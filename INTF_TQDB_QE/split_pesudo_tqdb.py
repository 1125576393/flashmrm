#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_file_size(file_path):
    """获取文件大小（GB）"""
    return os.path.getsize(file_path) / (1024 * 1024 * 1024)

def get_file_size_mb(file_path):
    """获取文件大小（MB）"""
    return os.path.getsize(file_path) / (1024 * 1024)

def get_total_rows(file_path):
    """快速获取文件总行数（不包括表头）"""
    logger.info("快速计算文件总行数...")
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        # 跳过表头
        next(f)
        # 使用大缓冲区读取
        buffer_size = 1024 * 1024 * 128  # 128MB缓冲区
        while True:
            buffer = f.read(buffer_size)
            if not buffer:
                break
            count += buffer.count('\n')
    return count

def split_large_csv_fast(input_file: str, output_prefix: str, max_file_size_mb: float = 24.0):
    """
    高效拆分超大型CSV文件（按文件大小拆分）
    
    参数:
        input_file: 输入文件路径
        output_prefix: 输出文件前缀
        max_file_size_mb: 每个输出文件的最大大小（MB），默认24MB
    """
    logger.info(f"开始拆分文件: {input_file}")
    file_size_gb = get_file_size(input_file)
    logger.info(f"文件大小: {file_size_gb:.2f} GB")
    logger.info(f"每个输出文件最大大小: {max_file_size_mb} MB")
    
    # 获取总行数
    total_rows = get_total_rows(input_file)
    logger.info(f"文件总行数: {total_rows}")
    
    # 读取表头
    with open(input_file, 'r', encoding='utf-8') as f:
        header = f.readline()
    
    # 输出文件列表
    output_files = []
    current_file_index = 1
    
    # 确保输出目录存在
    dir_path = os.path.dirname(output_prefix)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # 创建第一个输出文件
    current_output_file = f"{output_prefix}_part{current_file_index}.csv"
    output_files.append(current_output_file)
    current_writer = open(current_output_file, 'w', encoding='utf-8', buffering=1024 * 1024)  # 1MB缓冲区
    current_writer.write(header)
    
    # 打开输入文件，逐行读取
    logger.info("开始处理文件...")
    start_time = time.time()
    processed_rows = 0
    last_log_time = start_time
    
    with open(input_file, 'r', encoding='utf-8', buffering=1024 * 1024) as f:  # 1MB缓冲区
        # 跳过表头
        next(f)
        
        # 逐行处理
        for line in f:
            # 写入到当前文件
            current_writer.write(line)
            processed_rows += 1
            
            # 检查当前文件大小（每100行检查一次以提高效率）
            if processed_rows % 100 == 0:
                current_writer.flush()  # 确保数据写入磁盘
                current_file_size_mb = get_file_size_mb(current_output_file)
                
                # 如果当前文件大小超过限制，切换到新文件
                if current_file_size_mb >= max_file_size_mb:
                    # 关闭当前文件
                    current_writer.close()
                    logger.info(f"文件 {current_output_file} 已完成，大小: {current_file_size_mb:.2f} MB")
                    
                    # 创建新文件
                    current_file_index += 1
                    current_output_file = f"{output_prefix}_part{current_file_index}.csv"
                    output_files.append(current_output_file)
                    current_writer = open(current_output_file, 'w', encoding='utf-8', buffering=1024 * 1024)
                    current_writer.write(header)
            
            # 定期显示进度
            current_time = time.time()
            if current_time - last_log_time > 5:  # 每5秒记录一次进度
                elapsed = current_time - start_time
                rows_per_sec = processed_rows / elapsed if elapsed > 0 else 0
                remaining_rows = total_rows - processed_rows
                remaining_time = remaining_rows / rows_per_sec if rows_per_sec > 0 else 0
                
                logger.info(f"进度: {processed_rows}/{total_rows} 行 ({processed_rows/total_rows*100:.1f}%)")
                logger.info(f"速度: {rows_per_sec:.0f} 行/秒")
                logger.info(f"预计剩余时间: {remaining_time/60:.1f} 分钟")
                logger.info(f"已创建 {len(output_files)} 个输出文件")
                
                last_log_time = current_time
    
    # 关闭最后一个文件
    if not current_writer.closed:
        current_writer.close()
        logger.info(f"文件 {current_output_file} 已完成")
    
    # 记录结果
    logger.info("=" * 50)
    logger.info("拆分完成，文件列表:")
    total_output_size = 0
    for file in output_files:
        if os.path.exists(file):
            file_size = get_file_size_mb(file)
            total_output_size += file_size
            logger.info(f"  {file} - 大小: {file_size:.2f} MB")
        else:
            logger.warning(f"  {file} 未创建")
    
    logger.info(f"共生成 {len(output_files)} 个文件，总大小: {total_output_size:.2f} MB")
    return output_files

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='超大型CSV文件快速拆分工具（按文件大小拆分）')
    parser.add_argument('input_file', type=str, help='输入文件路径')
    parser.add_argument('output_prefix', type=str, help='输出文件前缀')
    parser.add_argument('--max_file_size_mb', type=float, default=24.0, 
                       help='每个输出文件的最大大小（MB），默认: 24.0')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        split_large_csv_fast(
            input_file=args.input_file,
            output_prefix=args.output_prefix,
            max_file_size_mb=args.max_file_size_mb
        )
    except Exception as e:
        logger.error(f"拆分过程中出错: {e}")
        raise
    
    elapsed = time.time() - start_time
    logger.info(f"总耗时: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分钟)")

if __name__ == "__main__":
    main()