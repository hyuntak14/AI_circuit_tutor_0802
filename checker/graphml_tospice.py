# graphml_to_spice_converter.py
import os
import glob
import networkx as nx
from networkx.readwrite import read_graphml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

class GraphMLToSpiceConverter:
    def __init__(self, input_dir=".", output_dir="./converted"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "spice"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        # CircuitSaver와 동일한 색상 매핑
        self.node_colors = {
            'VoltageSource': '#FF6B6B',  # 빨강
            'V-': '#0066FF',             # 파랑
            'Resistor': '#FF9500',       # 주황
            'Capacitor': '#8000FF',      # 보라
            'Diode': '#00FF00',          # 초록
            'LED': '#FFFF00',            # 노랑
            'IC': '#FF00FF',             # 마젠타
            'Junction': '#808080',       # 회색
            'Wire': '#95A5A6',           # 연결선
            'Unknown': '#BDC3C7'         # 미지의 컴포넌트
        }

    def normalize_component_class(self, cls):
        """컴포넌트 클래스 정규화"""
        if not cls:
            return 'Unknown'
        
        cls_map = {
            'VoltageSource': 'VoltageSource',
            'V+': 'VoltageSource',
            'V-': 'V-',  # V-는 접지로 별도 처리
            'Resistor': 'Resistor',
            'Capacitor': 'Capacitor',
            'Diode': 'Diode',
            'LED': 'LED',
            'IC': 'IC',
            'Junction': 'Junction',
            'Line_area': 'Wire',
            'Wire': 'Wire'
        }
        
        return cls_map.get(cls, cls)

    def create_node_mapping(self, graph):
        """노드 이름을 숫자로 매핑"""
        node_mapping = {}
        node_counter = 1
        
        # V- 노드들은 0 (접지)으로 매핑
        for node_name, data in graph.nodes(data=True):
            comp_class = self.normalize_component_class(data.get('type', ''))
            if comp_class == 'V-':
                node_mapping[node_name] = 0
        
        # 나머지 노드들은 순차적으로 번호 할당
        for node_name, data in graph.nodes(data=True):
            if node_name not in node_mapping:
                node_mapping[node_name] = node_counter
                node_counter += 1
        
        return node_mapping

    def get_connected_nodes(self, graph, node_name, node_mapping):
        """노드에 연결된 다른 노드들의 매핑된 번호 반환"""
        connected = []
        
        for u, v in graph.edges:
            if u == node_name:
                connected.append(node_mapping.get(v, v))
            elif v == node_name:
                connected.append(node_mapping.get(u, u))
        
        # 중복 제거 및 정렬
        return sorted(list(set(connected)))

    def generate_spice_netlist(self, graph, filename):
        """GraphML 그래프를 SPICE netlist로 변환"""
        spice_lines = []
        
        # 헤더 정보
        voltage_count = sum(1 for _, data in graph.nodes(data=True) 
                           if self.normalize_component_class(data.get('type', '')) == 'VoltageSource')
        
        spice_lines.append("* Multi-Power Circuit Netlist")
        spice_lines.append(f"* Generated with {voltage_count} power sources")
        spice_lines.append(f"* Converted from: {filename}")
        spice_lines.append("* ")
        
        # 노드 매핑 생성
        node_mapping = self.create_node_mapping(graph)
        
        # 컴포넌트 정의
        component_counter = {}
        
        for node_name, data in graph.nodes(data=True):
            comp_class = self.normalize_component_class(data.get('type', ''))
            
            if comp_class in ['Junction', 'V-', 'Wire']:
                continue  # 이런 노드들은 SPICE에서 제외
                
            # 컴포넌트 번호 생성
            if comp_class not in component_counter:
                component_counter[comp_class] = 1
            else:
                component_counter[comp_class] += 1
                
            comp_id = component_counter[comp_class]
            value = data.get('value', 0.0)
            
            # 연결된 노드들 찾기
            connected_nodes = self.get_connected_nodes(graph, node_name, node_mapping)
            
            # SPICE 라인 생성
            if comp_class == 'VoltageSource':
                if len(connected_nodes) >= 2:
                    spice_lines.append(f"V{comp_id} {connected_nodes[0]} {connected_nodes[1]} {value}")
                else:
                    spice_lines.append(f"V{comp_id} {connected_nodes[0] if connected_nodes else node_mapping[node_name]} 0 {value}")
                    
            elif comp_class == 'Resistor':
                if len(connected_nodes) >= 2:
                    spice_lines.append(f"R{comp_id} {connected_nodes[0]} {connected_nodes[1]} {value}")
                else:
                    spice_lines.append(f"R{comp_id} {connected_nodes[0] if connected_nodes else node_mapping[node_name]} 0 {value}")
                    
            elif comp_class == 'Capacitor':
                capacitor_value = f"{value}u" if value > 0 else "1u"
                if len(connected_nodes) >= 2:
                    spice_lines.append(f"C{comp_id} {connected_nodes[0]} {connected_nodes[1]} {capacitor_value}")
                else:
                    spice_lines.append(f"C{comp_id} {connected_nodes[0] if connected_nodes else node_mapping[node_name]} 0 {capacitor_value}")
                    
            elif comp_class == 'Diode':
                if len(connected_nodes) >= 2:
                    spice_lines.append(f"D{comp_id} {connected_nodes[0]} {connected_nodes[1]} DMOD")
                else:
                    spice_lines.append(f"D{comp_id} {connected_nodes[0] if connected_nodes else node_mapping[node_name]} 0 DMOD")
                    
            elif comp_class == 'LED':
                if len(connected_nodes) >= 2:
                    spice_lines.append(f"D{comp_id} {connected_nodes[0]} {connected_nodes[1]} LEDMOD")
                else:
                    spice_lines.append(f"D{comp_id} {connected_nodes[0] if connected_nodes else node_mapping[node_name]} 0 LEDMOD")
                    
            elif comp_class == 'IC':
                model = data.get('model', 'ua741')
                # IC 연결 정보 (단순화)
                conn_str = " ".join(map(str, connected_nodes)) if connected_nodes else str(node_mapping[node_name])
                spice_lines.append(f"X{comp_id} {conn_str} {model}")
        
        # 구분선과 모델 정의
        spice_lines.append("* ")
        spice_lines.append(".MODEL DMOD D")
        spice_lines.append(".MODEL LEDMOD D(IS=1E-12 N=2)")
        spice_lines.append(".END")
        
        return "\n".join(spice_lines)

    def draw_circuit_graph(self, graph, filename):
        """회로 그래프를 PNG 이미지로 저장"""
        if not graph.nodes():
            print(f"⚠️  {filename}: 빈 그래프입니다.")
            return
        
        # 그래프 레이아웃 설정
        plt.figure(figsize=(12, 8))
        
        # 스프링 레이아웃 사용 (더 나은 배치를 위해)
        pos = nx.spring_layout(graph, seed=42, k=2.0, iterations=50)
        
        # 노드 색상과 라벨 설정
        node_colors = []
        labels = {}
        
        for node_name, data in graph.nodes(data=True):
            comp_class = self.normalize_component_class(data.get('type', ''))
            node_colors.append(self.node_colors.get(comp_class, '#BDC3C7'))
            
            # 라벨 생성
            label = str(node_name)
            value = data.get('value', 0)
            
            if isinstance(value, (int, float)) and value != 0:
                # 단위 추가
                unit_map = {
                    'Resistor': 'Ω',
                    'VoltageSource': 'V',
                    'Capacitor': 'F'
                }
                unit = unit_map.get(comp_class, '')
                label += f"\n{value}{unit}"
            
            # 컴포넌트 타입도 표시
            if comp_class != 'Unknown':
                label += f"\n[{comp_class}]"
            
            labels[node_name] = label
        
        # 그래프 그리기
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=2000, alpha=0.9, edgecolors='black', linewidths=2)
        nx.draw_networkx_edges(graph, pos, edge_color='#2C3E50', 
                              width=2, alpha=0.8)
        nx.draw_networkx_labels(graph, pos, labels, font_size=8, font_weight='bold')
        
        # 엣지 라벨 (핀 정보가 있는 경우)
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            src_pin = data.get('source_pin', '')
            tgt_pin = data.get('target_pin', '')
            if src_pin or tgt_pin:
                edge_labels[(u, v)] = f"{src_pin}->{tgt_pin}"
        
        if edge_labels:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=6, font_color='red')
        
        # 제목과 통계 정보
        component_stats = self.get_component_stats(graph)
        title = f"Circuit: {os.path.splitext(filename)[0]}\n"
        title += f"Nodes: {len(graph.nodes())}, Edges: {len(graph.edges())}"
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        
        # 범례 생성
        used_components = set()
        for _, data in graph.nodes(data=True):
            comp_class = self.normalize_component_class(data.get('type', ''))
            used_components.add(comp_class)
        
        legend_elements = []
        for comp in sorted(used_components):
            color = self.node_colors.get(comp, '#BDC3C7')
            count = component_stats.get(comp, 0)
            legend_elements.append(
                patches.Patch(color=color, label=f"{comp} ({count})")
            )
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # 축 제거 및 레이아웃 조정
        plt.axis('off')
        plt.tight_layout()
        
        # PNG 파일로 저장
        base_name = os.path.splitext(filename)[0]
        png_path = os.path.join(self.output_dir, "images", f"{base_name}.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ PNG saved: {png_path}")

    def get_component_stats(self, graph):
        """컴포넌트 통계 계산"""
        stats = Counter()
        for _, data in graph.nodes(data=True):
            comp_class = self.normalize_component_class(data.get('type', ''))
            stats[comp_class] += 1
        return stats

    def convert_single_file(self, graphml_path):
        """단일 GraphML 파일 변환"""
        try:
            # GraphML 파일 읽기
            graph = read_graphml(graphml_path)
            filename = os.path.basename(graphml_path)
            base_name = os.path.splitext(filename)[0]
            
            print(f"🔄 Processing: {filename}")
            print(f"   Nodes: {len(graph.nodes())}, Edges: {len(graph.edges())}")
            
            # SPICE netlist 생성 및 저장
            spice_content = self.generate_spice_netlist(graph, filename)
            spice_path = os.path.join(self.output_dir, "spice", f"{base_name}.spice")
            
            with open(spice_path, 'w', encoding='utf-8') as f:
                f.write(spice_content)
            
            print(f"✅ SPICE saved: {spice_path}")
            
            # PNG 이미지 생성
            self.draw_circuit_graph(graph, filename)
            
            # 컴포넌트 통계 출력
            stats = self.get_component_stats(graph)
            if stats:
                print(f"   Components: {dict(stats)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {graphml_path}: {str(e)}")
            return False

    def convert_all_files(self):
        """현재 디렉토리의 모든 GraphML 파일 변환"""
        # GraphML 파일 찾기
        graphml_pattern = os.path.join(self.input_dir, "*.graphml")
        graphml_files = glob.glob(graphml_pattern)
        
        if not graphml_files:
            print("❌ GraphML 파일을 찾을 수 없습니다.")
            return
        
        print("=" * 60)
        print("🔧 GraphML to SPICE Converter")
        print("=" * 60)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Found {len(graphml_files)} GraphML files")
        print()
        
        # 각 파일 변환
        successful = 0
        failed = 0
        
        for graphml_path in graphml_files:
            if self.convert_single_file(graphml_path):
                successful += 1
            else:
                failed += 1
            print()
        
        # 결과 요약
        print("=" * 60)
        print("📊 CONVERSION SUMMARY")
        print("=" * 60)
        print(f"✅ Successfully converted: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📁 SPICE files saved to: {os.path.join(self.output_dir, 'spice')}")
        print(f"🖼️  PNG images saved to: {os.path.join(self.output_dir, 'images')}")
        print("=" * 60)

    def create_summary_report(self):
        """변환 결과 요약 보고서 생성"""
        spice_files = glob.glob(os.path.join(self.output_dir, "spice", "*.spice"))
        png_files = glob.glob(os.path.join(self.output_dir, "images", "*.png"))
        
        report_lines = []
        report_lines.append("# Circuit Conversion Report")
        report_lines.append(f"Generated on: {os.path.dirname(os.path.abspath(__file__))}")
        report_lines.append("")
        report_lines.append("## Summary")
        report_lines.append(f"- SPICE files created: {len(spice_files)}")
        report_lines.append(f"- PNG images created: {len(png_files)}")
        report_lines.append("")
        
        if spice_files:
            report_lines.append("## SPICE Files")
            for spice_file in sorted(spice_files):
                basename = os.path.basename(spice_file)
                report_lines.append(f"- {basename}")
        
        if png_files:
            report_lines.append("")
            report_lines.append("## PNG Images")
            for png_file in sorted(png_files):
                basename = os.path.basename(png_file)
                report_lines.append(f"- {basename}")
        
        # 보고서 저장
        report_path = os.path.join(self.output_dir, "conversion_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        print(f"📄 Report saved: {report_path}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert GraphML files to SPICE format and PNG images')
    parser.add_argument('--input', '-i', default='.', help='Input directory (default: current directory)')
    parser.add_argument('--output', '-o', default='./converted', help='Output directory (default: ./converted)')
    parser.add_argument('--report', '-r', action='store_true', help='Generate summary report')
    
    args = parser.parse_args()
    
    # 변환기 생성 및 실행
    converter = GraphMLToSpiceConverter(args.input, args.output)
    converter.convert_all_files()
    
    if args.report:
        converter.create_summary_report()


if __name__ == "__main__":
    main()