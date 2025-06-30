#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import warnings

class HassyFilmSlice:
    """用於分割 1x3 的 120 底片掃描 TIF 圖像的工具"""
    
    def __init__(self, input_path, output_dir=None, white_threshold=240, black_threshold=15):
        """
        初始化 HassyFilmSlice
        
        參數:
            input_path: 輸入 TIF 圖像的路徑
            output_dir: 輸出目錄，如果為 None，則使用輸入文件的目錄
        """
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"找不到輸入文件: {input_path}")
            
        if output_dir is None:
            self.output_dir = self.input_path.parent
        else:
            self.output_dir = Path(output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
            
        self.image = None
        self.frame_positions = []
        self.white_threshold = white_threshold
        self.black_threshold = black_threshold

    def load_image(self):
        """載入 TIF 圖像，保留原始格式和元數據"""
        try:
            # 使用 PIL 載入圖像，保留原始格式
            # original_img = Image.open(self.input_path)
            self.image = cv2.imread(self.input_path, cv2.IMREAD_UNCHANGED)
            self.frame_positions = [(0, self.image.shape[0], 0, self.image.shape[1])]

            print(f"成功載入圖像: {self.input_path}")
            print(f"圖像尺寸: {self.image.shape}")
            return True
        except Exception as e:
            print(f"載入圖像時出錯: {e}")
            return False
    
    def detect_frames(self, axis):
        """
        檢測底片框的位置
        
        參數:
            axis: 檢測軸，0 表示水平軸，1 表示垂直軸
        """
        if self.image is None or len(self.frame_positions) < 1:
            print("請先載入圖像")
            return False

        if self.image.dtype == np.uint16:
            # 16bits
            white_threshold = self.white_threshold*255
            black_threshold = self.black_threshold*255
            max_gray = 65535
        else:
            # 8bits
            white_threshold = self.white_threshold
            black_threshold = self.black_threshold
            max_gray = 255

        new_frame_positions = []
        for frame in self.frame_positions:
            cur_image = self.image[frame[0]:frame[1], frame[2]:frame[3]]

            # 轉換為灰度圖像進行處理，但不影響原始圖像
            if len(cur_image.shape) == 3:
                gray = cv2.cvtColor(cur_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = cur_image 

            # 使用閾值處理找出黑色分隔線(16-bits)
            _, thresh_black = cv2.threshold(gray, black_threshold, max_gray, cv2.THRESH_BINARY_INV)
            _, thresh_white = cv2.threshold(gray, white_threshold, max_gray, cv2.THRESH_BINARY)

            thresh = thresh_black + thresh_white

            # 水平投影分析，檢測黑色分隔線的位置
            h_proj = np.mean(thresh, axis=axis)
            
            # 找出峰值，這些峰值對應於分隔線的位置
            peaks = []
            threshold = np.max(h_proj) * 0.8  # 設定峰值檢測閾值

            print('threshold:', threshold)
            if threshold < max_gray * 0.5:
                # 这个方向上没有需要切分的
                new_frame_positions.append(frame)
                continue

            for i in range(1, len(h_proj) - 1):
                if h_proj[i] > threshold:
                    peaks.append(i)
            # 一般哈苏最多扫半格的 12x2张
            photo_min_size = len(h_proj) // 6 * 0.5

            peak_group = []
            group = []
            for p in peaks:
                if len(group) == 0 or (len(group) > 0 and abs(p - group[-1]) < photo_min_size):
                    group.append(p)
                else:
                    if len(group) > 0:
                        peak_group.append(group)
                    group = [p]
            if len(group) > 0:
                peak_group.append(group)

            peak_group = [int(np.mean(pg)) for pg in peak_group]
                
            # 根據分隔線計算三個框的位置
            # axis 0 表示水平軸，1 表示垂直軸
            size = self.image.shape[1-axis]
        
            peak_group = [0] + peak_group + [size]
            for idx, peak in enumerate(peak_group):
                if idx == 0:
                    continue
                if peak - peak_group[idx-1] > photo_min_size:
                    if axis == 1:
                        new_frame_positions.append((peak_group[idx-1], peak, frame[2], frame[3]))
                    else:
                        new_frame_positions.append((frame[0], frame[1], peak_group[idx-1], peak))
                
        print(f"檢測到的框位置: {new_frame_positions}")
        self.frame_positions = new_frame_positions

        return True
    
    def split_and_save(self):
        """分割圖像並保存為單獨的 TIF 文件，保持原始的高解析度和色深"""
        if not self.frame_positions:
            print("請先檢測框的位置")
            return False
        
        result_paths = []
        
        for i, (y0, y1, x0, x1) in enumerate(self.frame_positions):
            # 裁剪圖像
            frame = self.image[y0:y1, x0:x1]
            
            output_path = self.output_dir / f"{self.input_path.stem}_frame_{i+1}.tif"

            cv2.imwrite(output_path, frame) 
            
            result_paths.append(output_path)
            print(f"已保存框 {i+1}: {output_path}")
        
        return result_paths
    
    def process(self):
        """處理整個工作流程"""
        if not self.load_image():
            return False
        
        if not self.detect_frames(axis=1):
            return False
        if not self.detect_frames(axis=0):
            return False

        return self.split_and_save()


def process_directory(input_dir, output_dir=None):
    """
    處理指定目錄中的所有 TIF 文件
    
    參數:
        input_dir: 輸入目錄路徑
        output_dir: 輸出目錄路徑，如果為 None，則使用輸入目錄
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"錯誤: {input_dir} 不是有效的目錄")
        return False
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # 處理目錄中的所有文件和子目錄
    for item in input_dir.glob("*"):
        # 如果是目錄，則遞迴處理
        if item.is_dir():
            print(f"處理子目錄: {item}")
            sub_output_dir = output_dir / item.name if output_dir != input_dir else None
            s, f, sk = process_directory(item, sub_output_dir)
            success_count += s
            failed_count += f
            skipped_count += sk
        
        # 如果是 TIF 文件且文件名中不包含 'frame'，則進行處理
        elif item.is_file() and item.suffix.lower() in ['.tif', '.tiff', '.jpg', '.jpeg', '.png'] and 'frame' not in item.name.lower():
            print(f"處理文件: {item}")
            try:
                # 為每個文件創建對應的輸出目錄
                file_output_dir = output_dir
                
                # 處理文件
                splitter = HassyFilmSlice(item, file_output_dir)
                result = splitter.process()
                
                if result:
                    success_count += 1
                    print(f"成功處理: {item}")
                else:
                    failed_count += 1
                    print(f"處理失敗: {item}")
            except Exception as e:
                failed_count += 1
                print(f"處理 {item} 時發生錯誤: {e}")
                raise e
        else:
            if item.is_file() and item.suffix.lower() in ['.tif', '.tiff', '.jpg', '.jpeg', '.png'] and 'frame' in item.name.lower():
                print(f"跳過已處理的文件: {item}")
                skipped_count += 1
    
    print(f"目錄 {input_dir} 處理完成:")
    print(f"  成功: {success_count} 文件")
    print(f"  失敗: {failed_count} 文件")
    print(f"  跳過: {skipped_count} 文件")
    
    return success_count, failed_count, skipped_count


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='分割 Hasselblad X1/X5 扫描的整条胶片')
    parser.add_argument('input', help='输入图像(TIF/TIFF/JPG/JPEG/PNG)的路径或包含图像的目录')
    parser.add_argument('--white-threshold', type=int, default=240, help='白色阈值(0-255, 默认240)')
    parser.add_argument('--black-threshold', type=int, default=15, help='黑色阈值(0-255, 默认15)')
    parser.add_argument('-o', '--output', help='输出目录(默认与输入目录相同)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 判斷輸入是文件還是目錄
    if input_path.is_file():
        # 處理單個文件
        splitter = HassyFilmSlice(args.input, args.output, args.white_threshold, args.black_threshold)
        result = splitter.process()
        
        if result:
            print("處理完成!")
        else:
            print("處理失敗!")
    elif input_path.is_dir():
        # 處理整個目錄
        print(f"處理目錄: {input_path}")
        success_count, failed_count, skipped_count = process_directory(input_path, args.output)
        
        print("\n總結:")
        print(f"  成功處理: {success_count} 文件")
        print(f"  處理失敗: {failed_count} 文件")
        print(f"  已跳過: {skipped_count} 文件")
    else:
        print(f"錯誤: {input_path} 既不是文件也不是目錄")


if __name__ == "__main__":
    main()
