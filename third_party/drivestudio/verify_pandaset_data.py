import os
import glob

data_root = "data/pandaset/processed"
scenes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

print("="*80)
print(f"PandaSet Data Verification")
print("="*80)
print(f"\nTotal scenes: {len(scenes)}\n")

complete_scenes = []
issues = []

for scene in scenes:
    scene_path = os.path.join(data_root, scene)
    
    # 检查关键目录
    checks = {
        'images': len(glob.glob(f"{scene_path}/images/*.jpg")),
        'lidar': len(glob.glob(f"{scene_path}/lidar/*.bin")),
        'sky_masks': len(glob.glob(f"{scene_path}/sky_masks/*.png")),
        'ego_pose': len(glob.glob(f"{scene_path}/ego_pose/*.txt")),
        'dynamic_masks/all': len(glob.glob(f"{scene_path}/dynamic_masks/all/*.png")) if os.path.exists(f"{scene_path}/dynamic_masks/all") else 0,
        'instances/instances_info.json': os.path.exists(f"{scene_path}/instances/instances_info.json"),
    }
    
    # 判断是否完整 (80帧 × 6相机 = 480张图)
    is_complete = (
        checks['images'] >= 480 and 
        checks['lidar'] == 80 and
        checks['sky_masks'] >= 480 and
        checks['ego_pose'] == 80 and
        checks['instances/instances_info.json']
    )
    
    if is_complete:
        complete_scenes.append(scene)
    else:
        issues.append({
            'scene': scene,
            'checks': checks
        })

print(f"✅ Complete scenes: {len(complete_scenes)}/{len(scenes)}")

if issues:
    print(f"\n⚠️  Scenes with issues: {len(issues)}")
    for issue in issues[:5]:  # 只显示前5个
        print(f"\n  Scene {issue['scene']}:")
        for key, val in issue['checks'].items():
            if key == 'instances/instances_info.json':
                print(f"    {key}: {'✓' if val else '✗'}")
            else:
                expected = 480 if 'images' in key or 'masks' in key else 80
                status = '✓' if val >= expected else f'✗ ({val}/{expected})'
                print(f"    {key}: {status}")
else:
    print("\n🎉 All scenes are complete and ready for training!")

print("\n" + "="*80)
