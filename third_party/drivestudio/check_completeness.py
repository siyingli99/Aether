import os
import json

# 读取应处理的场景列表
with open('data/pandaset_example_scenes.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()[1:] if line.strip()]
    expected_scenes = [line.split(',')[0] for line in lines]

print(f"应处理场景总数: {len(expected_scenes)}")
print(f"{'='*80}")

# 检查每个场景的完整性
complete_scenes = []
incomplete_scenes = []
missing_scenes = []

for scene_id in expected_scenes:
    scene_dir = f'data/pandaset/processed/{scene_id}'
    
    if not os.path.exists(scene_dir):
        missing_scenes.append(scene_id)
        continue
    
    # 检查关键目录和文件
    checks = {
        'images': os.path.exists(f'{scene_dir}/images'),
        'lidar': os.path.exists(f'{scene_dir}/lidar'),
        'extrinsics': os.path.exists(f'{scene_dir}/extrinsics'),
        'intrinsics': os.path.exists(f'{scene_dir}/intrinsics'),
        'ego_pose': os.path.exists(f'{scene_dir}/ego_pose'),
        'dynamic_masks': os.path.exists(f'{scene_dir}/dynamic_masks'),
        'instances': os.path.exists(f'{scene_dir}/instances'),
    }
    
    # 统计文件数量
    img_count = len(os.listdir(f'{scene_dir}/images')) if checks['images'] else 0
    lidar_count = len(os.listdir(f'{scene_dir}/lidar')) if checks['lidar'] else 0
    
    # 检查instances JSON文件
    instances_complete = False
    if checks['instances']:
        info_file = f'{scene_dir}/instances/instances_info.json'
        frame_file = f'{scene_dir}/instances/frame_instances.json'
        instances_complete = os.path.exists(info_file) and os.path.exists(frame_file)
    
    # 判断是否完整（80帧 × 6个相机 = 480张图片，80个lidar文件）
    is_complete = (
        all(checks.values()) and 
        img_count >= 480 and 
        lidar_count == 80 and
        instances_complete
    )
    
    if is_complete:
        complete_scenes.append(scene_id)
    else:
        incomplete_scenes.append({
            'scene_id': scene_id,
            'images': img_count,
            'lidar': lidar_count,
            'checks': checks,
            'instances': instances_complete
        })

# 输出结果
print(f"\n✅ 完整场景: {len(complete_scenes)}/{len(expected_scenes)}")
print(f"⚠️  不完整场景: {len(incomplete_scenes)}")
print(f"❌ 缺失场景: {len(missing_scenes)}")

if incomplete_scenes:
    print(f"\n{'='*80}")
    print("不完整场景详情:")
    for item in incomplete_scenes[:10]:  # 只显示前10个
        print(f"\n场景 {item['scene_id']}:")
        print(f"  - 图片数: {item['images']} (期望: 480)")
        print(f"  - 点云数: {item['lidar']} (期望: 80)")
        print(f"  - Instances: {'✓' if item['instances'] else '✗'}")
        for key, val in item['checks'].items():
            if not val:
                print(f"  - 缺失目录: {key}")

if missing_scenes:
    print(f"\n{'='*80}")
    print(f"缺失场景 ({len(missing_scenes)}个):")
    print(", ".join(missing_scenes[:20]))

# 生成总结报告
print(f"\n{'='*80}")
print("处理总结:")
print(f"  总场景数: {len(expected_scenes)}")
print(f"  已完成: {len(complete_scenes)} ({len(complete_scenes)/len(expected_scenes)*100:.1f}%)")
print(f"  不完整: {len(incomplete_scenes)}")
print(f"  缺失: {len(missing_scenes)}")

if len(complete_scenes) == len(expected_scenes):
    print(f"\n🎉 所有场景处理完成！")
else:
    print(f"\n⚠️  有 {len(incomplete_scenes) + len(missing_scenes)} 个场景需要重新处理")
    
    # 创建需要重新处理的场景列表
    need_reprocess = [s['scene_id'] for s in incomplete_scenes] + missing_scenes
    if need_reprocess:
        with open('data/pandaset_incomplete_scenes.txt', 'w') as f:
            f.write("# scene_id, start_timestep, end_timestep\n")
            for scene_id in need_reprocess:
                f.write(f"{scene_id},0,-1\n")
        print(f"\n已创建重新处理列表: data/pandaset_incomplete_scenes.txt")

