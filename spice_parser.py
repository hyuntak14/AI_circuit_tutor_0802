# spice_parser.py
import re
from typing import List, Dict, Tuple, Optional

class SpiceParser:
    """SPICE 넷리스트를 파싱하여 회로 정보를 추출하는 클래스"""
    
    def __init__(self):
        self.components = []
        self.title = ""
        self.commands = []
        
    def parse_file(self, filepath: str) -> Dict:
        """SPICE 파일을 파싱하여 회로 정보를 반환"""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        return self.parse_lines(lines)
    
    def parse_lines(self, lines: List[str]) -> Dict:
        """SPICE 넷리스트 라인들을 파싱"""
        self.components = []
        self.commands = []
        
        # 첫 줄은 타이틀
        if lines:
            self.title = lines[0].strip()
        
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('*'):  # 주석 또는 빈 줄
                continue
                
            # 명령어 처리
            if line.startswith('.'):
                self.commands.append(line)
                continue
            
            # 컴포넌트 파싱
            parts = line.split()
            if not parts:
                continue
                
            name = parts[0]
            prefix = name[0].upper()
            
            # 컴포넌트 타입별 파싱
            if prefix == 'R':  # 저항
                if len(parts) >= 4:
                    self.components.append({
                        'name': name,
                        'type': 'Resistor',
                        'nodes': (int(parts[1]), int(parts[2])),
                        'value': self._parse_value(parts[3])
                    })
            elif prefix == 'C':  # 커패시터
                if len(parts) >= 4:
                    self.components.append({
                        'name': name,
                        'type': 'Capacitor',
                        'nodes': (int(parts[1]), int(parts[2])),
                        'value': self._parse_value(parts[3])
                    })
            elif prefix == 'L':  # 인덕터
                if len(parts) >= 4:
                    self.components.append({
                        'name': name,
                        'type': 'Inductor',
                        'nodes': (int(parts[1]), int(parts[2])),
                        'value': self._parse_value(parts[3])
                    })
            elif prefix == 'V':  # 전압원
                if len(parts) >= 4:
                    # DC 전압원 처리
                    if 'DC' in parts[3].upper() or parts[3].replace('.','').replace('-','').isdigit():
                        value = self._extract_dc_value(parts[3:])
                        self.components.append({
                            'name': name,
                            'type': 'VoltageSource',
                            'nodes': (int(parts[1]), int(parts[2])),
                            'value': value,
                            'source_type': 'DC'
                        })
            elif prefix == 'I':  # 전류원
                if len(parts) >= 4:
                    self.components.append({
                        'name': name,
                        'type': 'CurrentSource',
                        'nodes': (int(parts[1]), int(parts[2])),
                        'value': self._parse_value(parts[3])
                    })
            elif prefix == 'D':  # 다이오드
                if len(parts) >= 3:
                    self.components.append({
                        'name': name,
                        'type': 'Diode',
                        'nodes': (int(parts[1]), int(parts[2])),
                        'model': parts[3] if len(parts) > 3 else 'Dmodel'
                    })
            elif prefix == 'Q':  # BJT 트랜지스터
                if len(parts) >= 4:
                    self.components.append({
                        'name': name,
                        'type': 'BJT',
                        'nodes': (int(parts[1]), int(parts[2]), int(parts[3])),
                        'model': parts[4] if len(parts) > 4 else 'Qmodel'
                    })
            elif prefix == 'M':  # MOSFET
                if len(parts) >= 5:
                    self.components.append({
                        'name': name,
                        'type': 'MOSFET',
                        'nodes': (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])),
                        'model': parts[5] if len(parts) > 5 else 'Mmodel'
                    })
        
        return {
            'title': self.title,
            'components': self.components,
            'commands': self.commands
        }
    
    def _parse_value(self, value_str: str) -> float:
        """값 문자열을 파싱하여 숫자로 변환"""
        value_str = value_str.upper()
        
        # 단위 접미사 처리
        multipliers = {
            'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'M': 1e-3,
            'K': 1e3, 'U': 1e-6, 'N': 1e-9, 'P': 1e-12,
            'F': 1e-15
        }
        
        # 숫자와 단위 분리
        match = re.match(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([A-Z]*)', value_str)
        if match:
            number = float(match.group(1))
            unit = match.group(2)
            
            if unit in multipliers:
                return number * multipliers[unit]
            return number
        
        return 0.0
    
    def _extract_dc_value(self, parts: List[str]) -> float:
        """DC 값 추출"""
        value_str = ' '.join(parts)
        if 'DC' in value_str.upper():
            # DC 뒤의 값 추출
            match = re.search(r'DC\s+([-+]?\d*\.?\d+)', value_str, re.IGNORECASE)
            if match:
                return float(match.group(1))
        else:
            # 첫 번째 숫자를 DC 값으로 간주
            try:
                return float(parts[0])
            except:
                pass
        return 0.0