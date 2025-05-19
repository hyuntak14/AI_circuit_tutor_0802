"""
ErrorChecker 클래스를 정의하여 회로도 생성 후 다음과 같은 배선 오류를 검출합니다:
1. 단선(Open Circuit)
2. 단락(Short Circuit)
3. 플로팅 컴포넌트(Floating Component)
4. 고아 컴포넌트(Orphan Component)
5. 자기 자신으로의 루프(Loop on Same Net)
6. 다중 전원 소스 충돌(Multiple VoltageSource on One Net)
7. 극성 오류(Polarity Error)
8. 그라운드 루프(Ground Loop)
9. 와이어 크로스 연결 미감지(Unconnected Crossings)
10. 넷 이름 불일치(Net Name Mismatch)
11. 출력-출력 충돌(Output–Output Conflict)
12. 전압원 미검출(Missing Voltage Source)
13. 접지 미존재(No Ground Reference)
14. 컴포넌트 중복 연결(Duplicate Component Connection)

사용 예시:
    checker = ErrorChecker(components, nets, ground_nodes)
    errors = checker.run_all_checks()
    for err in errors:
        print(err)

components: List[dict], 각 dict는 {'name': str, 'class': str, 'nodes': tuple(node1, node2), 'value': Any, 'polarity': Optional[str]}
nets: Dict[node_id, List[component_names]]
"""
import networkx as nx
from collections import defaultdict

class ErrorChecker:
    def __init__(self, components, nets, ground_nodes=None):
        """
        :param components: [{'name', 'class', 'nodes': (n1, n2), 'value', 'polarity'}]
        :param nets: {node_id: [component_name, ...]}
        :param ground_nodes: Optional[Set[node_id]]: 정의된 접지 노드 집합
        """
        self.components = components
        self.nets = nets
        self.ground_nodes = set(ground_nodes or [])
        # VoltageSource 컴포넌트 리스트와 노드 집합 추출
        self.voltage_components = [c for c in components if c.get('class') == 'VoltageSource']
        self.voltage_nodes = set(n for comp in self.voltage_components for n in comp['nodes'])
        self.graph = self._build_graph()

    def _build_graph(self):
        """모든 노드를 그래프에 추가하여 NodeNotFound 방지"""
        G = nx.Graph()
        G.add_nodes_from(self.nets.keys())
        for comp in self.components:
            n1, n2 = comp['nodes']
            G.add_edge(n1, n2, component=comp)
        return G

    def _has_path(self, u, v):
        """노드 존재 여부 확인 후 경로 검사"""
        if u not in self.graph.nodes or v not in self.graph.nodes:
            return False
        return nx.has_path(self.graph, u, v)

    def detect_open_circuit(self):
        """전원 또는 접지에 연결되지 않은 컴포넌트 검출"""
        errors = []
        targets = self.ground_nodes | self.voltage_nodes
        for comp in self.components:
            n1, n2 = comp['nodes']
            if not any(self._has_path(n1, t) for t in targets) and \
               not any(self._has_path(n2, t) for t in targets):
                errors.append(f"Open circuit: component {comp['name']} disconnected from power or ground")
        return errors

    def detect_short_circuit(self):
        """전원 노드와 접지 노드가 직접 와이어로 연결된 경우 검출"""
        errors = []
        for u, v, data in self.graph.edges(data=True):
            comp = data['component']
            if comp.get('class') == 'Wire' and ((u in self.voltage_nodes and v in self.ground_nodes) or (v in self.voltage_nodes and u in self.ground_nodes)):
                errors.append(f"Short circuit: wire {comp['name']} connects voltage node {u} to ground node {v}")
        return errors

    def detect_floating_components(self):
        """
        소자 또는 와이어가 다른 어떤 소자/와이어와도 전기적으로 연결되지 않은 경우 검출
        → 그래프에서 edge(=component) 양 끝 노드의 degree가 모두 1일 때(자기 자신만) 떠다니는 것으로 간주
        """
        errors = []
        for comp in self.components:
            n1, n2 = comp['nodes']
            # 양끝 노드의 그래프 차수가 모두 1이면 이 comp만 연결된 것
            if self.graph.degree(n1) == 1 and self.graph.degree(n2) == 1:
                errors.append(
                    f"Floating component: {comp['name']} ({comp.get('class')}) is not connected to any other component or wire"
                )
        return errors


    def detect_orphan_components(self):
        """올바른 net 매핑이 없는 소자 검출"""
        errors = []
        for comp in self.components:
            n1, n2 = comp['nodes']
            if n1 not in self.nets or n2 not in self.nets:
                errors.append(f"Orphan component: {comp['name']} has invalid node mapping {comp['nodes']}")
        return errors

    def detect_self_loops(self):
        """같은 노드에 양 끝이 연결된 루프 검출"""
        errors = []
        for comp in self.components:
            n1, n2 = comp['nodes']
            if n1 == n2:
                errors.append(f"Self-loop: component {comp['name']} connected between same node {n1}")
        return errors

    def detect_mult_voltage_sources(self):
        """하나의 net에 둘 이상의 VoltageSource 연결 검출 (수정된 버전)"""
        errors = []
        
        # 각 net에 연결된 서로 다른 전압원들을 추적
        vs_by_net = defaultdict(set)  # set을 사용하여 중복 제거
        
        for comp in self.voltage_components:
            vs_name = comp['name']
            for net in comp['nodes']:
                vs_by_net[net].add(vs_name)
        
        # 같은 net에 2개 이상의 서로 다른 전압원이 연결된 경우만 에러
        for net, vs_names in vs_by_net.items():
            if len(vs_names) > 1:
                errors.append(f"Multiple voltage sources on net {net}: {list(vs_names)}")
        
        return errors

    def detect_polarity_errors(self):
        """다이오드/콘덴서 극성 오류 검출"""
        errors = []
        for comp in self.components:
            if comp.get('polarity') in ('+', '-'):
                plus_idx = 0 if comp['polarity'] == '+' else 1
                node = comp['nodes'][plus_idx]
                if node in self.ground_nodes:
                    errors.append(f"Polarity error: positive pin of {comp['name']} on ground node {node}")
        return errors

    def detect_ground_loops(self):
        """접지 루프 검출"""
        errors = []
        sub = self.graph.subgraph(self.ground_nodes)
        if nx.number_of_edges(sub) > len(self.ground_nodes) - 1:
            errors.append("Ground loop detected among ground nodes")
        return errors

    def detect_unconnected_crossings(self):
        """와이어 교차 미연결 검출 (미구현)"""
        return []

    def detect_net_name_mismatch(self):
        """물리적 넷 ID 분리 검출 (미구현)"""
        return []

    def detect_output_conflicts(self):
        """여러 output driver가 같은 net 연결된 경우 검출"""
        errors = []
        out_by_net = defaultdict(list)
        for comp in self.components:
            if comp.get('class') == 'OutputDriver':
                for n in comp['nodes']:
                    out_by_net[n].append(comp['name'])
        for node, names in out_by_net.items():
            if len(names) > 1:
                errors.append(f"Output-output conflict on net {node}: {names}")
        return errors

    def detect_missing_voltage_source(self):
        """부하에 전압원 연결 없는 경우 검출"""
        errors = []
        for comp in self.components:
            if comp.get('class') not in ('VoltageSource', 'Wire'):
                if not any(self._has_path(n, vs) for n in comp['nodes'] for vs in self.voltage_nodes):
                    errors.append(f"Missing voltage source for component {comp['name']}")
        return errors

    def detect_no_ground_reference(self):
        """접지 정의 없는 경우 검출"""
        if not self.ground_nodes:
            return ["No ground reference defined"]
        return []

    def detect_duplicate_connections(self):
        """중복 연결 검출"""
        errors = []
        seen = set()
        for comp in self.components:
            key = (comp['name'], tuple(sorted(comp['nodes'])))
            if key in seen:
                errors.append(f"Duplicate connection for component {comp['name']} nodes {comp['nodes']}")
            seen.add(key)
        return errors

    def run_all_checks(self):
        """모든 오류 검출 실행"""
        checks = [
            self.detect_open_circuit,
            self.detect_short_circuit,
            self.detect_floating_components,
            #self.detect_orphan_components,
            self.detect_self_loops,
            self.detect_mult_voltage_sources,
            self.detect_polarity_errors,
            self.detect_ground_loops,
            self.detect_unconnected_crossings,
            self.detect_net_name_mismatch,
            self.detect_output_conflicts,
            self.detect_missing_voltage_source,
            self.detect_no_ground_reference,
            self.detect_duplicate_connections,
        ]
        errors = []
        for fn in checks:
            errors.extend(fn())
        return errors
