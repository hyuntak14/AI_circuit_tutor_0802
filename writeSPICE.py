import pandas as pd
import numpy as np

# code = '''
# * SPICE
# '''

# '''
# * SPICE DP #1

# V  1 0 DC 12
# R1 1 2 10k
# R2 2 3 10k
# R3 3 0 10k

# .op
# .tran 0 10ms 0ms 0.1ms
# .print v(1) v(2) v(3)
# .end

# '''

def toSPICE(circuit: pd.DataFrame, voltage: int, path: str):
    # 인덱스용 num_idx 컬럼 추가
    num_c = 0
    for idx, row in circuit.iterrows():
        circuit.loc[idx, "num_idx"] = num_c
        num_c += 1
    circuit["num_idx"] = circuit["num_idx"].astype(np.int32)
    circuit["layer"]   = circuit["layer"].astype(np.int32)

    # 최소·최대 레이어 동적 계산
    min_layer = int(circuit["layer"].min())
    max_layer = int(circuit["layer"].max())

    start_network = circuit[circuit["layer"] == min_layer]
    end_network   = circuit[circuit["layer"] == max_layer]

    code = "* SPICE \n\n"

    # V source: 노드 번호는 (max_layer-1, min_layer)
    line  = f"V {max_layer - 1} {min_layer} DC {voltage}\n"

    # prev_point 초기화
    prev_point = min_layer
    # start_idx 는 첫 소자 인덱스 추출
    start_idx = int(start_network.iloc[-1].num_idx)

    code += line + "\n"
    node_count = 0
    nn = 1

    # 나머지 소자들을 순차적으로 넷리스트에 추가
    for idx, row in circuit.iloc[start_idx + 1 :].iterrows():
        next_point = int(row.layer)
        if row["class"] == "Line":
            continue

        line  = f"{row['name']} {next_point} {prev_point} {int(row['value'])}\n"

        # 레이어(노드) 변경 시 새로운 노드 카운트
        try:
            next_comp = circuit.iloc[nn + 1]
            if int(next_comp.layer) != next_point:
                node_count += 1
                prev_point = next_point
            code += line
            nn += 1
        except IndexError:
            print("End of circuit")

    # OP 및 트랜션트 명령
    code += "\n"
    code += ".op\n"
    code += ".tran 0 10ms 0ms 0.1ms\n"

    # 노드 전압 출력
    code += ".print "
    for n in range(1, node_count + 1):
        code += f"v({n}) "
    code += "\n.end\n"

    # 파일 쓰기
    with open(path, "w") as f:
        f.write(code)


if __name__ == "__main__":
    circuit_1 = pd.read_json("./circuit_detected_1.json")
    circuit_2 = pd.read_json("./circuit_detected_2.json")
    circuit_3 = pd.read_json("./circuit_detected_3.json")

    print(circuit_1)
