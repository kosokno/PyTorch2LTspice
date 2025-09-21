"""
PyTorch2LTspice
===============

Utility to convert PyTorch nn.Sequential models (Linear, Activations, RNNCell/GRUCell/LSTMCell)
into LTspice-compatible subcircuits.

This script extracts weights/biases from a trained model and generates an .SUBCKT netlist
with behavioral sources and .machine-based recurrent cells. Intended for power electronics
control and reinforcement learning research where PyTorch-trained policies are deployed
inside LTspice simulations.

Author: github.com/kosokno
License: MIT

Change Log:
2025-06-10:
- Initial release.

2025-09-13:
- Multi-output support.
  - `export_model_to_ltspice` and `generate_ltspice_subckt` now take
    `output_ports=None` and auto-assign NNOUT* when not provided (removed `output_port`).
- Recurrent support: Added nn.RNNCell, nn.GRUCell, nn.LSTMCell.
  - Auto ports: inputs use NNIN* (+ CLK if any Cell exists; Cell states are internal).
  - Outputs: NNOUT*; Cell state nets are internal only.
- Activation: Added TANH handling.

Notes:
- No default input voltage sources are emitted.
  - Parent circuits must drive NNIN* explicitly (legacy "default sources" were removed in earlier versions).
- Recurrent Cells with .machine + CLK gating:
  - If any Cell layer exists, `CLK` is automatically added as a subcircuit input.
  - Each Cell uses a three-state (LO/LATCH/HI) .machine with LH*/LC* latches to capture previous outputs in the LATCH phase;
    no external SAMPLEHOLD subcircuits are emitted.
- Feature node selection for Cells:
  - For RNN/GRU/LSTM cells, the feature vector X is formed from the first `input_size` NNIN* nodes only.
  - Control/state pins `CLK`, `HIN*`, `CIN*`, `LH*`, `LC*` are excluded from X.
  - A strict length check enforces `len(features) == input_size` to catch wiring mistakes early.
- GRU candidate state n and r-gate:
  - The reset gate r is applied element-wise AFTER forming (W_hh_n h + b_hh_n):
    ñ = tanh(W_in x + b_in + r ⊙ (W_hh_n h + b_hh_n)).
- Multi-cell support and port naming:
  - With any Cell present, per-cell namespaces are used for internal states; externally only `CLK` and NNIN*/NNOUT* are exposed.
- CLK thresholds:
  - .rule uses V(CLK) > .5 / > .9 as margin to avoid metastability around latch transitions.
- State node initialization:
  - HOUT*/LH* (and COUT*/LC* for LSTM) have 1k pull-downs to guarantee a defined 0 V initial value.
"""

import torch
import torch.nn as nn
import numpy as np

def extract_layers(model):
    """
    Extract layer info from an nn.Sequential model.
    - Linear: export weights (W) and bias (b)
    - Activations: ReLU / Sigmoid / Tanh
    - Cells: nn.RNNCell / nn.GRUCell / nn.LSTMCell (weights/biases, input_size, hidden_size, etc.)
    Only nn.Sequential is supported.
    """ 
    layers_info = []
    if isinstance(model, nn.Sequential):
        for module in model:
            if isinstance(module, nn.Linear):
                W = module.weight.detach().cpu().numpy()
                b = module.bias.detach().cpu().numpy()
                layers_info.append({"type": "linear", "W": W, "b": b})
            elif isinstance(module, nn.ReLU):
                layers_info.append({"type": "activation", "act": "ReLU"})
            elif isinstance(module, nn.Sigmoid):
                layers_info.append({"type": "activation", "act": "Sigmoid"})
            elif isinstance(module, nn.Tanh):
                layers_info.append({"type": "activation", "act": "Tanh"})
            elif isinstance(module, nn.RNNCell):
                W_ih = module.weight_ih.detach().cpu().numpy()
                W_hh = module.weight_hh.detach().cpu().numpy()
                b_ih = module.bias_ih.detach().cpu().numpy() if module.bias_ih is not None else np.zeros(W_ih.shape[0])
                b_hh = module.bias_hh.detach().cpu().numpy() if module.bias_hh is not None else np.zeros(W_hh.shape[0])
                layers_info.append({
                    "type": "rnncell",
                    "W_ih": W_ih,
                    "W_hh": W_hh,
                    "b_ih": b_ih,
                    "b_hh": b_hh,
                    "input_size": int(module.input_size),
                    "hidden_size": int(module.hidden_size),
                    "nonlin": getattr(module, "nonlinearity", "tanh"),
                })
            elif isinstance(module, nn.GRUCell):
                W_ih = module.weight_ih.detach().cpu().numpy()
                W_hh = module.weight_hh.detach().cpu().numpy()
                b_ih = module.bias_ih.detach().cpu().numpy() if module.bias_ih is not None else np.zeros(W_ih.shape[0])
                b_hh = module.bias_hh.detach().cpu().numpy() if module.bias_hh is not None else np.zeros(W_hh.shape[0])
                layers_info.append({
                    "type": "grucell",
                    "W_ih": W_ih,
                    "W_hh": W_hh,
                    "b_ih": b_ih,
                    "b_hh": b_hh,
                    "input_size": int(module.input_size),
                    "hidden_size": int(module.hidden_size),
                })
            elif isinstance(module, nn.LSTMCell):
                W_ih = module.weight_ih.detach().cpu().numpy()
                W_hh = module.weight_hh.detach().cpu().numpy()
                b_ih = module.bias_ih.detach().cpu().numpy() if module.bias_ih is not None else np.zeros(W_ih.shape[0])
                b_hh = module.bias_hh.detach().cpu().numpy() if module.bias_hh is not None else np.zeros(W_hh.shape[0])
                layers_info.append({
                    "type": "lstmcell",
                    "W_ih": W_ih,
                    "W_hh": W_hh,
                    "b_ih": b_ih,
                    "b_hh": b_hh,
                    "input_size": int(module.input_size),
                    "hidden_size": int(module.hidden_size),
                })
            else:
                # Ignore unsupported modules
                pass
        return layers_info
    else:
        raise ValueError("Currently, only nn.Sequential models are supported.")

def generate_dot_product_expression(input_nodes, weights, neuron_index):
    """
    Generates a dot product expression for the LTspice behavioral source.
    For example: "V(NNIN1)*(-0.179081)+V(NNIN2)*(-0.068428)+..."
    """
    terms = []
    for i, node in enumerate(input_nodes):
        w_val = weights[neuron_index, i]
        # Force uppercase for node names.
        terms.append(f"V({node.upper()})*({w_val:.6f})")
    expr = "+".join(terms)
    return expr.upper()

def generate_ltspice_subckt(layers_info, subckt_name="NETLISTSUBCKT", input_ports=None, output_ports=None):
    """
    Generates an LTspice subcircuit netlist from the extracted layer information.
    For linear layers, creates behavioral sources that compute the dot product 
    (input*weight + bias). For activation layers, creates behavioral sources implementing
    ReLU / Sigmoid / Tanh. The final output is connected to the external output ports.
    
    If input_ports is not provided, they are auto-generated in the format "NNIN1, NNIN2, ..." 
    based on the input dimension of the first linear layer.

    If at least one recurrent Cell layer (RNNCell/GRUCell/LSTMCell) exists, a `CLK` pin is added
    to the subcircuit inputs automatically. The Cell layers are implemented with `.machine` blocks
    that use a three-state LO/LATCH/HI sequence, capturing the previous outputs on LH*/LC* latch
    nodes during the LATCH phase and updating new values only when the machine reaches HI.
    No external SAMPLEHOLD elements are emitted.
    """
    netlist_lines = []

    # Analyze recurrent cells for multi-cell support
    cell_layers = [
        {"idx": i+1, "type": ly["type"], "hidden_size": ly["hidden_size"]}
        for i, ly in enumerate([l for l in layers_info if l["type"] in ("rnncell", "grucell", "lstmcell")])
    ]
    total_cells = len(cell_layers)
    # Use clearer flag name: any recurrent cell exists
    has_any_cell = (total_cells > 0)
    
    # Auto-generate input ports if not provided
    if input_ports is None:
        # Determine base external inputs from the first layer that defines input size
        input_ports = None
        for layer in layers_info:
            if layer["type"] == "linear":
                in_dim = layer["W"].shape[1]
                input_ports = [f"NNIN{i+1}" for i in range(in_dim)]
                break
            elif layer["type"] in ("rnncell", "grucell", "lstmcell"):
                in_dim = layer["input_size"]
                input_ports = [f"NNIN{i+1}" for i in range(in_dim)]
                break
        if input_ports is None:
            input_ports = ["NNIN1"]

        # If any recurrent cell exists anywhere, add only the CLK pin (states are internal nets)
        if has_any_cell:
            input_ports += ["CLK"]
    
    # Initialize current nodes as the input ports (converted to uppercase)
    current_nodes = [node.upper() for node in input_ports]

    # Helper to select only feature nodes (exclude control/state like CLK/HIN*/CIN*)
    def _feature_nodes(nodes):
        nn = []
        for n in nodes:
            u = str(n).upper()
            if u == "CLK":
                continue
            if u.startswith("HIN"):
                continue
            if u.startswith("CIN"):
                continue
            if u.startswith("LH"):
                continue
            if u.startswith("LC"):
                continue
            nn.append(u)
        return nn
    linear_layer_count = 0
    activation_layer_count = 0
    cell_layer_index = 0   # counts recurrent cells for unique naming

    # Process each layer
    for layer in layers_info:
        if layer["type"] == "linear":
            linear_layer_count += 1
            W = layer["W"]
            b = layer["b"]
            out_dim = W.shape[0]
            in_dim = W.shape[1]
            # Use only the inputs this Linear expects. If more nodes are present
            # (e.g., because HIN*/CIN* were appended as ports), take the first
            # in_dim nodes which correspond to the original NNIN* or prior layer outputs.
            if len(current_nodes) < in_dim:
                raise ValueError(
                    f"Linear layer expects {in_dim} inputs but only {len(current_nodes)} nodes are available."
                )
            in_nodes = current_nodes[:in_dim]
            new_nodes = []
            netlist_lines.append(f"* LAYER {linear_layer_count}: LINEAR")
            for j in range(out_dim):
                # Define unique internal node names.
                node_name = f"L{linear_layer_count}_{j+1}".upper()
                new_nodes.append(node_name)
                dot_expr = generate_dot_product_expression(in_nodes, W, j)
                # Build expression using parentheses
                expr = f"({dot_expr}+({b[j]:.6f}))".upper()
                # Use the parameter "V=" in the behavioral voltage source line.
                netlist_lines.append(f"B{linear_layer_count}_{j+1} {node_name} 0 V={expr}")
            current_nodes = new_nodes
        elif layer["type"] == "activation":
            activation_layer_count += 1
            new_nodes = []
            netlist_lines.append(f"* ACTIVATION LAYER {activation_layer_count}: {layer['act'].upper()}")
            for j, old_node in enumerate(current_nodes):
                node_name = f"L_ACT{activation_layer_count}_{j+1}".upper()
                new_nodes.append(node_name)
                if layer["act"].upper() == "RELU":
                    # Use standard LTspice if() function with uppercase letters.
                    expr = f"(IF(V({old_node})>0,V({old_node}),0))".upper()
                elif layer["act"].upper() == "SIGMOID":
                    # Use standard logistic function
                    expr = f"(1/(1+EXP(-V({old_node}))))".upper()
                elif layer["act"].upper() == "TANH":
                    # Use LTspice built-in TANH
                    expr = f"(TANH(V({old_node})))".upper()
                else:
                    expr = f"(V({old_node}))".upper()
                netlist_lines.append(f"B_ACT{activation_layer_count}_{j+1} {node_name} 0 V={expr}")
            current_nodes = new_nodes
            last_semantics = None
        elif layer["type"] == "rnncell":
            netlist_lines.append("* RNNCell (.machine gated; inline compute)")
            cell_layer_index += 1
            in_dim = layer["input_size"]
            h_dim = layer["hidden_size"]
            _feats = _feature_nodes(current_nodes)
            if len(_feats) != in_dim:
                raise ValueError(f"Cell input_size={in_dim} but feature nodes={len(_feats)}")
            X = _feats[:in_dim]
            W_ih = layer["W_ih"]; W_hh = layer["W_hh"]
            b_ih = layer["b_ih"]; b_hh = layer["b_hh"]
            act = str(layer.get("nonlin", "tanh")).lower()
            out_nodes = []
            lh_nodes = []
            for j in range(h_dim):
                out_node = f"HOUT{cell_layer_index}_{j+1}".upper()
                latch_node = f"LH{cell_layer_index}_{j+1}".upper()
                out_nodes.append(out_node)
                lh_nodes.append(latch_node)
                netlist_lines.append(f"R_RNN{cell_layer_index}_{j+1} {out_nodes[j]} 0 1k")
                netlist_lines.append(f"R_RNN{cell_layer_index}_LH{j+1} {lh_nodes[j]} 0 1k")
            netlist_lines.append(".machine")
            netlist_lines.append(".state LO 0")
            netlist_lines.append(".state LATCH 1")
            netlist_lines.append(".state HI 2")
            netlist_lines.append(".rule LO LATCH V(CLK)>.5")
            netlist_lines.append(".rule LATCH HI V(CLK)>.9")
            netlist_lines.append(".rule * LO V(CLK)<.5")
            for j in range(h_dim):
                netlist_lines.append(f".output ({lh_nodes[j]}) IF((state==1), V({out_nodes[j]}), V({lh_nodes[j]}))")
            for j in range(h_dim):
                a_terms_x = '+'.join([f"V({X[i]})*({W_ih[j,i]:.6f})" for i in range(in_dim)])
                a_terms_h = '+'.join([f"V({lh_nodes[i]})*({W_hh[j,i]:.6f})" for i in range(h_dim)])
                bias_val = float(b_ih[j]) + float(b_hh[j])
                if a_terms_x and a_terms_h:
                    a_sum = f"{a_terms_x}+{a_terms_h}+({bias_val:.6f})"
                elif a_terms_x:
                    a_sum = f"{a_terms_x}+({bias_val:.6f})"
                elif a_terms_h:
                    a_sum = f"{a_terms_h}+({bias_val:.6f})"
                else:
                    a_sum = f"({bias_val:.6f})"
                if act == "relu":
                    next_h = f"(IF(({a_sum})>0,({a_sum}),0))"
                else:
                    next_h = f"(TANH({a_sum}))"
                netlist_lines.append(f".output ({out_nodes[j]}) IF((state==2), {next_h}, V({out_nodes[j]}))")
            netlist_lines.append(".endmachine")
            current_nodes = [n.upper() for n in out_nodes]
            saved_h_nodes = current_nodes[:]
            last_semantics = "h_only"
        elif layer["type"] == "grucell":
            netlist_lines.append("* GRUCell (.machine gated; inline compute; r_i precompute)")
            cell_layer_index += 1
            in_dim = layer["input_size"]
            H = layer["hidden_size"]

            # Use processed features for X; exclude CLK and latched state nets
            _feats = _feature_nodes(current_nodes)
            if len(_feats) != in_dim:
                raise ValueError(f"Cell input_size={in_dim} but feature nodes={len(_feats)}")
            X = _feats[:in_dim]

            W_ih = layer["W_ih"]; W_hh = layer["W_hh"]
            b_ih = layer["b_ih"]; b_hh = layer["b_hh"]
            Wi_r, Wi_z, Wi_n = W_ih[0:H,:],   W_ih[H:2*H,:],   W_ih[2*H:3*H,:]
            Wh_r, Wh_z, Wh_n = W_hh[0:H,:],   W_hh[H:2*H,:],   W_hh[2*H:3*H,:]
            bi_r, bi_z, bi_n = b_ih[0:H],     b_ih[H:2*H],     b_ih[2*H:3*H]
            bh_r, bh_z, bh_n = b_hh[0:H],     b_hh[H:2*H],     b_hh[2*H:3*H]

            out_nodes = []
            lh_nodes = []
            for j in range(H):
                node_name = f"HOUT{cell_layer_index}_{j+1}".upper()
                latch_name = f"LH{cell_layer_index}_{j+1}".upper()
                out_nodes.append(node_name)
                lh_nodes.append(latch_name)
                netlist_lines.append(f"R_GRU{cell_layer_index}_{j+1} {node_name} 0 1k")
                netlist_lines.append(f"R_GRU{cell_layer_index}_LH{j+1} {latch_name} 0 1k")

            r_nodes = [f"GRU{cell_layer_index}_R{i+1}" for i in range(H)]
            z_nodes = [f"GRU{cell_layer_index}_Z{i+1}" for i in range(H)]
            nx_nodes = [f"GRU{cell_layer_index}_NX{i+1}" for i in range(H)]
            hh_nodes = [f"GRU{cell_layer_index}_HH{i+1}" for i in range(H)]
            nh_nodes = [f"GRU{cell_layer_index}_NH{i+1}" for i in range(H)]
            n_nodes = [f"GRU{cell_layer_index}_N{i+1}" for i in range(H)]

            for i in range(H):
                netlist_lines.append(f"R_GRU{cell_layer_index}_R{i+1} {r_nodes[i]} 0 1k")
            for j in range(H):
                netlist_lines.append(f"R_GRU{cell_layer_index}_Z{j+1} {z_nodes[j]} 0 1k")
                netlist_lines.append(f"R_GRU{cell_layer_index}_NX{j+1} {nx_nodes[j]} 0 1k")
                netlist_lines.append(f"R_GRU{cell_layer_index}_HH{j+1} {hh_nodes[j]} 0 1k")
                netlist_lines.append(f"R_GRU{cell_layer_index}_NH{j+1} {nh_nodes[j]} 0 1k")
                netlist_lines.append(f"R_GRU{cell_layer_index}_N{j+1} {n_nodes[j]} 0 1k")
            netlist_lines.append(".machine")
            netlist_lines.append(".state LO 0")
            netlist_lines.append(".state LATCH 1")
            netlist_lines.append(".state HI 2")
            netlist_lines.append(".rule LO LATCH V(CLK)>.5")
            netlist_lines.append(".rule LATCH HI V(CLK)>.9")
            netlist_lines.append(".rule * LO V(CLK)<.5")
            for j in range(H):
                netlist_lines.append(f".output ({lh_nodes[j]}) IF((state==1), V({out_nodes[j]}), V({lh_nodes[j]}))")

            for i in range(H):
                ar_x = '+'.join([f"V({X[p]})*({Wi_r[i,p]:.6f})" for p in range(in_dim)]) if in_dim>0 else ''
                ar_h = '+'.join([f"V({lh_nodes[q]})*({Wh_r[i,q]:.6f})" for q in range(H)]) if H>0 else ''
                ar_bias = f"({float(bi_r[i] + bh_r[i]):.6f})"
                if ar_x and ar_h:
                    ar_sum = f"{ar_x}+{ar_h}+{ar_bias}"
                elif ar_x:
                    ar_sum = f"{ar_x}+{ar_bias}"
                elif ar_h:
                    ar_sum = f"{ar_h}+{ar_bias}"
                else:
                    ar_sum = f"{ar_bias}"
                rexp = f"(1/(1+EXP(-({ar_sum}))))"
                netlist_lines.append(f".output ({r_nodes[i]}) IF((state==2), {rexp}, V({r_nodes[i]}))")

            for j in range(H):
                az_x = '+'.join([f"V({X[p]})*({Wi_z[j,p]:.6f})" for p in range(in_dim)]) if in_dim>0 else ''
                az_h = '+'.join([f"V({lh_nodes[q]})*({Wh_z[j,q]:.6f})" for q in range(H)]) if H>0 else ''
                az_bias = f"({float(bi_z[j] + bh_z[j]):.6f})"
                if az_x and az_h:
                    az_sum = f"{az_x}+{az_h}+{az_bias}"
                elif az_x:
                    az_sum = f"{az_x}+{az_bias}"
                elif az_h:
                    az_sum = f"{az_h}+{az_bias}"
                else:
                    az_sum = f"{az_bias}"
                zexpr = f"(1/(1+EXP(-({az_sum}))))"
                netlist_lines.append(f".output ({z_nodes[j]}) IF((state==2), {zexpr}, V({z_nodes[j]}))")

            for j in range(H):
                nx_x = '+'.join([f"V({X[p]})*({Wi_n[j,p]:.6f})" for p in range(in_dim)]) if in_dim>0 else ''
                nx_bias = f"({float(bi_n[j]):.6f})"
                if nx_x:
                    nx_sum = f"{nx_x}+{nx_bias}"
                else:
                    nx_sum = f"{nx_bias}"
                netlist_lines.append(f".output ({nx_nodes[j]}) IF((state==2), ({nx_sum}), V({nx_nodes[j]}))")

            for j in range(H):
                hh_terms = '+'.join([f"({Wh_n[j,i]:.6f})*V({lh_nodes[i]})" for i in range(H)]) if H>0 else ''
                hh_bias  = f"({float(bh_n[j]):.6f})"
                if hh_terms:
                    hh_sum = f"{hh_terms}+{hh_bias}"
                else:
                    hh_sum = f"{hh_bias}"
                netlist_lines.append(f".output ({hh_nodes[j]}) IF((state==2), ({hh_sum}), V({hh_nodes[j]}))")
                netlist_lines.append(f".output ({nh_nodes[j]}) IF((state==2), ( V({r_nodes[j]}) * V({hh_nodes[j]}) ), V({nh_nodes[j]}))")

            for j in range(H):
                netlist_lines.append(f".output ({n_nodes[j]}) IF((state==2), TANH( V({nx_nodes[j]}) + V({nh_nodes[j]}) ), V({n_nodes[j]}))")

            for j in range(H):
                update = f"( (1 - V({z_nodes[j]})) * V({n_nodes[j]}) + V({z_nodes[j]}) * V({lh_nodes[j]}) )"
                netlist_lines.append(f".output ({out_nodes[j]}) IF((state==2), {update}, V({out_nodes[j]}))")

            netlist_lines.append(".endmachine")

            current_nodes = [n.upper() for n in out_nodes]
            saved_h_nodes = current_nodes[:]
            last_semantics = "h_only"
        elif layer["type"] == "lstmcell":
            netlist_lines.append("* LSTMCell (.machine gated; inline compute)")
            cell_layer_index += 1
            in_dim = layer["input_size"]
            H = layer["hidden_size"]
            # Use processed features for X (exclude CLK and latched state nets)
            _feats = _feature_nodes(current_nodes)
            if len(_feats) != in_dim:
                raise ValueError(f"Cell input_size={in_dim} but feature nodes={len(_feats)}")
            X = _feats[:in_dim]
            W_ih = layer["W_ih"]; W_hh = layer["W_hh"]
            b_ih = layer["b_ih"]; b_hh = layer["b_hh"]
            Wi_i, Wi_f, Wi_g, Wi_o = W_ih[0:H,:], W_ih[H:2*H,:], W_ih[2*H:3*H,:], W_ih[3*H:4*H,:]
            Wh_i, Wh_f, Wh_g, Wh_o = W_hh[0:H,:], W_hh[H:2*H,:], W_hh[2*H:3*H,:], W_hh[3*H:4*H,:]
            bi_i, bi_f, bi_g, bi_o = b_ih[0:H], b_ih[H:2*H], b_ih[2*H:3*H], b_ih[3*H:4*H]
            bh_i, bh_f, bh_g, bh_o = b_hh[0:H], b_hh[H:2*H], b_hh[2*H:3*H], b_hh[3*H:4*H]
            c_nodes = []
            h_nodes = []
            lc_nodes = []
            lh_nodes = []
            for j in range(H):
                c_name = f"COUT{cell_layer_index}_{j+1}".upper()
                h_name = f"HOUT{cell_layer_index}_{j+1}".upper()
                lc_name = f"LC{cell_layer_index}_{j+1}".upper()
                lh_name = f"LH{cell_layer_index}_{j+1}".upper()
                c_nodes.append(c_name)
                h_nodes.append(h_name)
                lc_nodes.append(lc_name)
                lh_nodes.append(lh_name)
                netlist_lines.append(f"R_LSTM{cell_layer_index}_C{j+1} {c_name} 0 1k")
                netlist_lines.append(f"R_LSTM{cell_layer_index}_H{j+1} {h_name} 0 1k")
                netlist_lines.append(f"R_LSTM{cell_layer_index}_LC{j+1} {lc_name} 0 1k")
                netlist_lines.append(f"R_LSTM{cell_layer_index}_LH{j+1} {lh_name} 0 1k")
            netlist_lines.append(".machine")
            netlist_lines.append(".state LO 0")
            netlist_lines.append(".state LATCH 1")
            netlist_lines.append(".state HI 2")
            netlist_lines.append(".rule LO LATCH V(CLK)>.5")
            netlist_lines.append(".rule LATCH HI V(CLK)>.9")
            netlist_lines.append(".rule * LO V(CLK)<.5")
            for j in range(H):
                netlist_lines.append(f".output ({lh_nodes[j]}) IF((state==1), V({h_nodes[j]}), V({lh_nodes[j]}))")
                netlist_lines.append(f".output ({lc_nodes[j]}) IF((state==1), V({c_nodes[j]}), V({lc_nodes[j]}))")
            for j in range(H):
                ai = "+".join([f"V({X[i]})*({Wi_i[j,i]:.6f})" for i in range(in_dim)]) + "+" + "+".join([f"V({lh_nodes[i]})*({Wh_i[j,i]:.6f})" for i in range(H)]) + f"+({float(bi_i[j]+bh_i[j]):.6f})"
                af = "+".join([f"V({X[i]})*({Wi_f[j,i]:.6f})" for i in range(in_dim)]) + "+" + "+".join([f"V({lh_nodes[i]})*({Wh_f[j,i]:.6f})" for i in range(H)]) + f"+({float(bi_f[j]+bh_f[j]):.6f})"
                ag = "+".join([f"V({X[i]})*({Wi_g[j,i]:.6f})" for i in range(in_dim)]) + "+" + "+".join([f"V({lh_nodes[i]})*({Wh_g[j,i]:.6f})" for i in range(H)]) + f"+({float(bi_g[j]+bh_g[j]):.6f})"
                ao = "+".join([f"V({X[i]})*({Wi_o[j,i]:.6f})" for i in range(in_dim)]) + "+" + "+".join([f"V({lh_nodes[i]})*({Wh_o[j,i]:.6f})" for i in range(H)]) + f"+({float(bi_o[j]+bh_o[j]):.6f})"
                iexpr = f"(1/(1+EXP(-({ai}))))"
                fexpr = f"(1/(1+EXP(-({af}))))"
                gexpr = f"(TANH({ag}))"
                oexpr = f"(1/(1+EXP(-({ao}))))"
                cnext = f"(({fexpr})*V({lc_nodes[j]}) + ({iexpr})*({gexpr}))"
                hnext = f"(({oexpr})*TANH({cnext}))"
                netlist_lines.append(f".output ({c_nodes[j]}) IF((state==2), {cnext}, V({c_nodes[j]}))")
                netlist_lines.append(f".output ({h_nodes[j]}) IF((state==2), {hnext}, V({h_nodes[j]}))")
            netlist_lines.append(".endmachine")
            current_nodes = [n.upper() for n in h_nodes]
            saved_h_nodes = current_nodes[:]
            saved_c_nodes = [n.upper() for n in c_nodes]
            last_semantics = "lstm_h"
        else:
            pass

    # Determine output ports and build header
    final_count = len(current_nodes)
    if output_ports is None:
        output_ports = [f"NNOUT{i+1}" for i in range(final_count)]
    out_count = len(output_ports)
    header_ports = output_ports
    header = f".SUBCKT {subckt_name} " + " ".join(input_ports) + (" " + " ".join(header_ports) if header_ports else "")
    # Insert header at top of file
    netlist_lines.insert(0, header)

    # Connect the final internal node(s) to the external output port(s) using behavioral sources.
    if final_count == 0:
        netlist_lines.append("* No final nodes to connect")
    elif final_count == 1 and out_count == 1:
        final_node = current_nodes[0]
        netlist_lines.append(f"* Connect final internal node {final_node} to external output {output_ports[0]}")
        netlist_lines.append(f"B_OUT {output_ports[0]} 0 V=V({final_node})")
    else:
        # Multi-output case: require the number of output ports to match the number of final nodes
        if out_count != final_count:
            raise ValueError(
                f"Output port count ({out_count}) does not match model output dimension ({final_count}). "
                f"Pass a list of {final_count} names to output_ports."
            )
        netlist_lines.append("* Connect final internal nodes to multiple external outputs")
        for idx, (node, outp) in enumerate(zip(current_nodes, output_ports), start=1):
            netlist_lines.append(f"B_OUT{idx} {outp} 0 V=V({node})")
    # Add alias outputs for hidden state (HOUT*) if defined
    # States are internal and updated via the .machine; no alias B-sources are emitted.
    netlist_lines.append(f".ENDS {subckt_name}")
    return "\n".join(netlist_lines)

def export_model_to_ltspice(model, filename="MODEL_SUBCKT.SP", subckt_name="NETLISTSUBCKT", input_ports=None, output_ports=None, verbose=True):
    """
    Extracts parameters from an nn.Sequential PyTorch model and exports an LTspice subcircuit
    netlist to a file. The file is written in ASCII encoding.
    - If output_ports is None, NNOUT* ports are auto-assigned; mismatch with model output size raises ValueError.
    """
    layers_info = extract_layers(model)
    netlist = generate_ltspice_subckt(layers_info, subckt_name, input_ports, output_ports=output_ports)
    with open(filename, "w", encoding="ascii") as f:
        f.write(netlist)
    if verbose:
        print(f"Exported model to LTspice subcircuit netlist in '{filename}'.")

if __name__ == "__main__":
    # Example 1: MLP (Linear -> ReLU -> Linear -> ReLU -> Linear)
    MLP_model = nn.Sequential(
        nn.Linear(19, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )
    MLP_model.eval()
    export_model_to_ltspice(MLP_model, filename="TEST_MODEL_MLP.SP", subckt_name="TESTMLP")

    # Example 2: GRUCell -> Linear -> Tanh
    GRU_model = nn.Sequential(
        nn.GRUCell(input_size=5, hidden_size=4),
        nn.Linear(4, 1),
        nn.Tanh()
    )
    GRU_model.eval()
    export_model_to_ltspice(GRU_model, filename="TEST_MODEL_GRUCELL.SP", subckt_name="TESTGRUCELL")
    
    # Example 3: LSTMCell -> Linear -> Tanh
    LSTM_model = nn.Sequential(
        nn.LSTMCell(input_size=5, hidden_size=4),
        nn.Linear(4, 1),
        nn.Tanh()
    )
    LSTM_model.eval()
    export_model_to_ltspice(LSTM_model, filename="TEST_MODEL_LSTMCELL.SP", subckt_name="TESTLSTMCELL")

    # Example 4: RNNCell -> Linear -> Tanh
    RNN_model = nn.Sequential(
        nn.RNNCell(input_size=5, hidden_size=4),
        nn.Linear(4, 1),
        nn.Tanh()
    )
    RNN_model.eval()
    export_model_to_ltspice(RNN_model, filename="TEST_MODEL_RNNCELL.SP", subckt_name="TESTRNNCELL")

    # Example 5: Linear -> ReLU -> GRUCell -> Linear -> Tanh
    LinGRU_model = nn.Sequential(
        nn.Linear(7, 32),
        nn.ReLU(),
        nn.GRUCell(input_size=32, hidden_size=32),
        nn.Linear(32, 1),
        nn.Tanh()
    )
    LinGRU_model.eval()
    export_model_to_ltspice(LinGRU_model, filename="TEST_MODEL_LINGRUCELL.SP", subckt_name="TESTLINGRUCELL")

    # Example 6: Linear -> ReLU -> LSTMCell -> Linear -> Tanh
    LinLSTM_model = nn.Sequential(
        nn.Linear(7, 32),
        nn.ReLU(),
        nn.LSTMCell(input_size=32, hidden_size=32),
        nn.Linear(32, 1),
        nn.Tanh()
    )
    LinLSTM_model.eval()
    export_model_to_ltspice(LinLSTM_model, filename="TEST_MODEL_LINLSTMCELL.SP", subckt_name="TESTLINLSTMCELL")

    # Example 7: Linear -> ReLU -> LSTMCell -> LSTMCell -> Linear -> Tanh
    MultiLSTM_model = nn.Sequential(
        nn.Linear(7, 16),
        nn.ReLU(),
        nn.LSTMCell(input_size=16, hidden_size=16),
        nn.LSTMCell(input_size=16, hidden_size=16),
        nn.Linear(16, 1),
        nn.Tanh()
    )
    MultiLSTM_model.eval()
    export_model_to_ltspice(MultiLSTM_model, filename="TEST_MODEL_MULTILSTMCELL.SP", subckt_name="TESTMULTILSTMCELL")


