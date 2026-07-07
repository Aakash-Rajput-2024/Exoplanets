import os
import glob

def fix_paths(directory, old_str, new_str):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') or file.endswith('.ipynb'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if old_str in content:
                        new_content = content.replace(old_str, new_str)
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"Fixed: {filepath}")
                except Exception as e:
                    print(f"Could not process {filepath}: {e}")

if __name__ == "__main__":
    src_dir = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets/src"
    old_path = "/Users/aakashrajput/MachineLearning/Exoplanets"
    new_path = "d:/Exoplanets/Neural Posterior Estimation/Exoplanets"
    
    # Also replace backslashes if needed, though python handles forward slashes fine in Windows
    fix_paths(src_dir, old_path, new_path)
    print("Done fixing paths!")
