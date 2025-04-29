# writeSPICE.py
import pandas as pd
import numpy as np

def toSPICE(circuit: pd.DataFrame, voltage: int, path: str):
    circuit = circuit.copy()
    circuit['value'] = circuit['value'].fillna(0)

    code = "* SPICE\n\n"
    # 전원 소스: GND는 node1 of first comp, V+는 node2 of last comp
    start = f"V1 {circuit.iloc[-1]['node2_n']} {circuit.iloc[0]['node1_n']} DC {voltage}\n\n"
    code += start
    for _, row in circuit.iterrows():
        n1 = int(row['node1_n'])
        n2 = int(row['node2_n'])
        name = row['name']
        val = row['value']
        cls = row['class']
        if cls == 'Resistor':
            code += f"{name} {n1} {n2} {int(val)}\n"
        elif cls == 'Capacitor':
            code += f"{name} {n1} {n2} Cmodel\n"
        elif cls == 'Diode':
            code += f"{name} {n1} {n2} Dmodel\n"
        elif cls == 'LED':
            code += f"{name} {n1} {n2} LEDmodel\n"
        elif cls == 'IC':
            code += f"X{name} {n1} {n2} Umodel\n"
        else:
            code += f"{name} {n1} {n2} {val}\n"
    code += "\n.op\n.tran 0 10ms 0ms 0.1ms\n.end\n"

    with open(path, 'w') as f:
        f.write(code)

if __name__ == "__main__":
    # 간단 테스트
    df = pd.DataFrame([
        {'name':'R1','class':'Resistor','value':100,'layer':0},
        {'name':'D1','class':'Diode','value':0,'layer':1},
    ])
    toSPICE(df, 5, "test.spice")
    print(open("test.spice").read())
