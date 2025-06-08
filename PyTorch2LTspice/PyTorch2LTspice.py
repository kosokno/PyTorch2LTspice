import torch
import torch.nn as nn
import numpy as np

def extract_layers(model):
    """
    Extracts layer information from an nn.Sequential model.
    For each Linear layer, extracts the weight (W) and bias (b).
    For activation layers, records the type (ReLU or Sigmoid).
    Currently, only nn.Sequential models are supported.
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

def generate_ltspice_subckt(layers_info, subckt_name="NETLISTSUBCKT", input_ports=None, output_port="NNOUT1"):
    """
    Generates an LTspice subcircuit netlist from the extracted layer information.
    For linear layers, creates behavioral sources that compute the dot product 
    (input*weight + bias). For activation layers, creates behavioral sources implementing
    the activation function (ReLU or Sigmoid). The final output is connected to the external
    output port.
    
    If input_ports is not provided, they are auto-generated in the format "NNIN1, NNIN2, ..." 
    based on the input dimension of the first linear layer.
    
    (Note: The default voltage sources for actor inputs have been removed,
    so the parent circuit must now drive NNIN1 ... NNIN19 or leave them floating.)
    """
    netlist_lines = []
    
    # Auto-generate input ports if not provided (based on first linear layer's input dimension)
    if input_ports is None:
        for layer in layers_info:
            if layer["type"] == "linear":
                in_dim = layer["W"].shape[1]
                input_ports = [f"NNIN{i+1}" for i in range(in_dim)]
                break
    # Subcircuit header with external output.
    header = f".SUBCKT {subckt_name} " + " ".join(input_ports) + f" {output_port}"
    netlist_lines.append(header)
    netlist_lines.append("")
    
    # (The default voltage source lines have been removed.)
    
    # Initialize current nodes as the input ports (converted to uppercase)
    current_nodes = [node.upper() for node in input_ports]
    linear_layer_count = 0
    activation_layer_count = 0

    # Process each layer
    for layer in layers_info:
        if layer["type"] == "linear":
            linear_layer_count += 1
            W = layer["W"]
            b = layer["b"]
            out_dim = W.shape[0]
            new_nodes = []
            netlist_lines.append(f"* LAYER {linear_layer_count}: LINEAR")
            for j in range(out_dim):
                # Define unique internal node names.
                node_name = f"L{linear_layer_count}_{j+1}".upper()
                new_nodes.append(node_name)
                dot_expr = generate_dot_product_expression(current_nodes, W, j)
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
                    expr = f"(1/(1+EXP(-V({old_node}))))".upper()
                else:
                    expr = f"(V({old_node}))".upper()
                netlist_lines.append(f"B_ACT{activation_layer_count}_{j+1} {node_name} 0 V={expr}")
            current_nodes = new_nodes
        else:
            pass

    # Connect the final internal node to the external output using a behavioral source.
    if len(current_nodes) >= 1:
        final_node = current_nodes[0]
    else:
        final_node = ""
    netlist_lines.append(f"* Connect final internal node {final_node} to external output {output_port}")
    netlist_lines.append(f"B_OUT {output_port} 0 V=V({final_node})")
    netlist_lines.append(f".ENDS {subckt_name}")
    return "\n".join(netlist_lines)

def export_model_to_ltspice(model, filename="MODEL_SUBCKT.SP", subckt_name="NETLISTSUBCKT", input_ports=None, output_port="NNOUT1", verbose=True):
    """
    Extracts parameters from an nn.Sequential PyTorch model and exports an LTspice subcircuit
    netlist to a file. The file is written in ASCII encoding.
    """
    layers_info = extract_layers(model)
    netlist = generate_ltspice_subckt(layers_info, subckt_name, input_ports, output_port)
    with open(filename, "w", encoding="ascii") as f:
        f.write(netlist)
    if verbose:
        print(f"Exported model to LTspice subcircuit netlist in '{filename}'.")

if __name__ == "__main__":
    # Define a simple test model using nn.Sequential for compatibility.
    test_model = nn.Sequential(
        nn.Linear(19, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
        # Optionally, add a Sigmoid activation if required:
        # nn.Sigmoid()
    )
    test_model.eval()
    # Export the model to an LTspice subcircuit netlist with external output "NNOUT1"
    # and auto-generated input ports "NNIN1, NNIN2, ..., NNIN19".
    export_model_to_ltspice(test_model, filename="TEST_MODEL_SUBCKT.SP", subckt_name="TESTACTORSUBCKT")
