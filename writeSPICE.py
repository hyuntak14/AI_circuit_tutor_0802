# writeSPICE.py
import pandas as pd
import numpy as np

def toSPICE(circuit: pd.DataFrame, voltage: int, path: str):
    # 결측치는 0으로 대체
    circuit['value'] = circuit['value'].fillna(0)

    # 인덱스용 num_idx, layer 타입 보정
    for idx, row in circuit.iterrows():
        circuit.loc[idx, "num_idx"] = idx
    circuit["num_idx"] = circuit["num_idx"].astype(np.int32)
    circuit["layer"]   = circuit["layer"].astype(np.int32)

    # SPICE 헤더
    min_layer = int(circuit["layer"].min())
    max_layer = int(circuit["layer"].max())
    code = "* SPICE\n\n"
    code += f"V {max_layer-1} {min_layer} DC {voltage}\n\n"

    prev_point = min_layer
    start_idx  = int(circuit[circuit["layer"] == min_layer].iloc[-1].num_idx)
    node_count = 0
    nn         = 1

    # 넷리스트 작성
    for idx, row in circuit.iloc[start_idx + 1 :].iterrows():
        next_point = int(row.layer)
        if row["class"] == "Line":
            continue

        cls = row["class"]
        if   cls == "Resistor":
            val = int(row["value"])
            line = f"{row['name']} {next_point} {prev_point} {val}\n"
        elif cls == "Capacitor":
            line = f"{row['name']} {next_point} {prev_point} Cmodel\n"
        elif cls == "Diode":
            line = f"{row['name']} {next_point} {prev_point} Dmodel\n"
        elif cls == "LED":
            line = f"{row['name']} {next_point} {prev_point} LEDmodel\n"
        elif cls == "IC":
            line = f"X{row['name']} {next_point} {prev_point} Umodel\n"
        else:
            line = f"{row['name']} {next_point} {prev_point} {row['value']}\n"

        # 노드 전환 체크
        try:
            next_comp = circuit.iloc[nn + 1]
            if int(next_comp.layer) != next_point:
                node_count += 1
                prev_point = next_point
            code += line
            nn += 1
        except IndexError:
            print("End of circuit")

    # OP/TRAN, 출력
    code += "\n.op\n"
    code += ".tran 0 10ms 0ms 0.1ms\n"
    code += ".print "
    for n in range(1, node_count + 1):
        code += f"v({n}) "
    code += "\n.end\n"

    # 파일 쓰기
    with open(path, "w") as f:
        f.write(code)

if __name__ == "__main__":
    # 간단 테스트
    df = pd.DataFrame([
        {'name':'R1','class':'Resistor','value':100,'layer':0},
        {'name':'D1','class':'Diode','value':0,'layer':1},
    ])
    toSPICE(df, 5, "test.spice")
    print(open("test.spice").read())
