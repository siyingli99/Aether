#!/usr/bin/env python3
"""
修复 pandaset_sourceloader.py 中 load_objects 方法
确保在不加载 SMPL 时也初始化 smpl_human_all 属性
"""

file_path = "datasets/pandaset/pandaset_sourceloader.py"

# 读取文件
with open(file_path, 'r') as f:
    lines = f.readlines()

# 找到 load_objects 方法中的 if self.data_cfg.load_smpl: 这一行
# 在这一行之后添加 else 分支
new_lines = []
modified = False

i = 0
while i < len(lines):
    line = lines[i]
    new_lines.append(line)
    
    # 找到 "if self.data_cfg.load_smpl:" 这一行
    if 'if self.data_cfg.load_smpl:' in line and not modified:
        # 获取这一行的缩进级别
        indent = len(line) - len(line.lstrip())
        
        # 跳过 if 代码块内的所有内容
        i += 1
        while i < len(lines):
            current_line = lines[i]
            current_indent = len(current_line) - len(current_line.lstrip())
            
            # 如果遇到与 if 同级别的代码（非注释、非空行），说明 if 块结束
            if (current_indent <= indent and 
                current_line.strip() and 
                not current_line.strip().startswith('#')):
                # 在这里插入 else 块（在当前行之前）
                else_block = [
                    ' ' * indent + 'else:\n',
                    ' ' * (indent + 4) + '# SMPL loading disabled, initialize empty dict\n',
                    ' ' * (indent + 4) + 'self.smpl_human_all = {}\n',
                    ' ' * (indent + 4) + 'logger.info("SMPL loading is disabled, initialized empty smpl_human_all")\n',
                ]
                new_lines.extend(else_block)
                modified = True
                # 添加当前行（if 块后的代码）
                new_lines.append(current_line)
                break
            else:
                # 继续添加 if 块内的内容
                new_lines.append(current_line)
                i += 1
        
        if i >= len(lines):
            # 如果 if 块是方法的最后部分，在文件末尾添加 else
            else_block = [
                '\n',
                ' ' * indent + 'else:\n',
                ' ' * (indent + 4) + '# SMPL loading disabled, initialize empty dict\n',
                ' ' * (indent + 4) + 'self.smpl_human_all = {}\n',
                ' ' * (indent + 4) + 'logger.info("SMPL loading is disabled, initialized empty smpl_human_all")\n',
            ]
            new_lines.extend(else_block)
            modified = True
    
    i += 1

if modified:
    # 写回文件
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    print("✅ Successfully modified pandaset_sourceloader.py")
    print("   Added else block to initialize smpl_human_all when load_smpl is False")
else:
    print("❌ Could not find the target line to modify")
    print("   Please check if the file structure has changed")
    exit(1)

