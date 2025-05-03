"""
音乐谱面加载模块，负责加载和解析音乐谱面数据。
"""
import json

class MusicSheetLoader:
    """音乐谱面加载类，用于从JSON文件加载音乐谱面数据。"""
    
    def __init__(self):
        """初始化音乐谱面加载器。"""
        self.beat_data = []
        self.keypoint_names = []
        self.keypoint_indices = []
        self.json_data = {}  # 添加json_data属性
    
    def load_music_sheet(self, json_path):
        """
        从JSON文件加载音乐谱面数据。
        
        Args:
            json_path: 包含音乐谱面数据的JSON文件路径
            
        Returns:
            bool: 如果成功加载数据则返回True，否则返回False
        """
        try:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)
                
            # 提取节拍数据
            self.beat_data = self.json_data.get('beats', [])
            
            # 提取关键点信息
            self.keypoint_names = self.json_data.get('keypoint_names', [])
            self.keypoint_indices = self.json_data.get('keypoint_indices', [])
            
            # 验证数据是否有效
            if len(self.beat_data) > 0:
                print(f"Successfully loaded music sheet: {json_path}")
                print(f"Total beats: {len(self.beat_data)}")
                print(f"Keypoint names: {self.keypoint_names}")
                return True
            else:
                print("Warning: No valid beat data found in JSON file")
                return False
                
        except Exception as e:
            print(f"Error loading music sheet: {str(e)}")
            self.beat_data = []
            self.keypoint_names = []
            self.keypoint_indices = []
            self.json_data = {}  # 重置json_data
            return False
    
    def get_beat_at_time(self, elapsed_time):
        """
        获取指定时间点应该出现的节拍。
        
        Args:
            elapsed_time: 从游戏开始到现在经过的时间（秒）
            
        Returns:
            list: 在当前时间应该生成的节拍数据，如果没有则返回空列表
        """
        beats_to_spawn = []
        
        for beat in self.beat_data:
            beat_time = beat.get('time', 0)
            
            # 如果节拍时间与当前时间匹配（在一定误差范围内）
            if abs(elapsed_time - beat_time) < 0.05:  # 50毫秒误差容许
                beats_to_spawn.append(beat)
        
        return beats_to_spawn
    
    def has_beat_data(self):
        """
        检查是否成功加载了节拍数据。
        
        Returns:
            bool: 如果有节拍数据则返回True，否则返回False
        """
        return len(self.beat_data) > 0