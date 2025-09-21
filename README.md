# PyTorch2LTspice

**PyTorch2LTspice** converts PyTorch neural network models into LTspice-compatible subcircuits (`.subckt`).  
By combining it with [LTspicePowerSim](https://github.com/kosokno/LTspicePowerSim), users can implement AI-based controllers directly in power electronics circuits such as DC-DC converters, inverters, and motor drivers.  

This repository also provides example code where **a neural network controls the PWM of a BUCK regulator, trained with behavior imitation and PPO**.


![Overview](./img/PyTorch2LTspice.png)

---

## 📌 Features

- Converts PyTorch `nn.Sequential` models to LTspice-compatible `.subckt` format
- Supported layers:
  - Linear:
    - `nn.Linear`
  - Activations:
    - `nn.ReLU`
    - `nn.Sigmoid`
    - `nn.Tanh`
  - Cells:
    - `nn.RNNCell`
    - `nn.GRUCell`
    - `nn.LSTMCell`
- Outputs a netlist using behavioral voltage sources (`B` elements)
- Recurrent cells are implemented with `.machine` blocks (LO/LATCH/HI states, CLK pin auto-added)
- Auto-generates LTspice node names (`NNIN1`, `NNIN2`, ..., `NNOUT1`, ...)
- Easy integration into LTspice testbenches
- Several example models are included:
  - MLP: `Linear → ReLU → Linear → ReLU → Linear → Sigmoid`
  - GRUCell: `GRUCell → Linear → Tanh`
  - LSTMCell: `LSTMCell → Linear → Tanh`
  - RNNCell: `RNNCell → Linear → Tanh`
  - Hybrid: `Linear → ReLU → (GRUCell/LSTMCell) → Linear → Tanh`
  - Multi-cell LSTM: stacked `LSTMCell` layers

---
## 🧠 Motivation

Neural networks trained in Python (with PyTorch) can now be exported and tested directly in LTspice circuit simulations.  
This allows for:
- Closed-loop simulation with NN controllers 
- Verification of inference logic inside switching power supplies
- Observation of behavior under nonlinear and dynamic conditions

---

## ✨ Training Example

### NN Controlled Voltage Mode Buck (WIP)
![Overview](./img/NN_BUCK_VM.png)


---

## 🚀 How to Use

### 1. Define a model in PyTorch

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(20, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
model.eval()
```

### 2. Export as LTspice `.subckt` file

```python
from PyTorch2LTspice import export_model_to_ltspice

export_model_to_ltspice(
    model,
    filename="TEST_MODEL_SUBCKT.SP",
    subckt_name="TESTACTORSUBCKT"
)
```

---

## 📂 Output Example

The resulting LTspice subcircuit will look like:

```
.SUBCKT TESTACTORSUBCKT NNIN1 NNIN2 ... NNIN20 NNOUT1
* LAYER 1: LINEAR
B1_1 L1_1 0 V=V(NNIN1)*(-0.179081)+V(NNIN2)*(-0.068428)+...
...
* ACTIVATION LAYER 1: RELU
B_ACT1_1 L_ACT1_1 0 V=(IF(V(L1_1)>0,V(L1_1),0))
...
B_OUT NNOUT1 0 V=V(L_ACT2_1)
.ENDS TESTACTORSUBCKT
```

You can include it in your LTspice schematic with `.INCLUDE TEST_MODEL_SUBCKT.SP`, and wire it to your simulated environment.

---

## 📎 Dependencies

- Python 3.7+
- PyTorch
- NumPy

Install with:

```bash
pip install torch numpy
```

---

## 📄 License

MIT License

---

## 🧩 Related Projects

- 🔗 [LTspicePowerSim](https://github.com/kosokno/LTspicePowerSim.git):  
  A Simulink-like power electronics simulation environment built on LTspice,

