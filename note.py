import os
import csv
import pandas as pd
from mp_api.client import MPRester

os.chdir("/home/edgpu/edgpu27/cgcnn_test/hands_on/train_sample")

api_key = "JItsGMnvTQt1mIkIduA3rorAHL3SDPBE"

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

with MPRester(api_key) as mpr:
    # Get summary documents for O-*-*-*-*-* chemical system
    print("Retrieving SummaryDoc documents...")
    summary_docs = mpr.materials.summary.search(
        chemsys="O-*-*-*-*-*",
        fields=["material_id", "composition", "band_gap"]
    )
    mpids = [doc.material_id for doc in summary_docs]

    print(f"Number of materials: {len(mpids)}")

    # Save materials_info.csv
    print("Saving materials_info.csv...")
    with open('materials_info.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Material ID", "Composition", "Band Gap"])
        for doc in summary_docs:
            writer.writerow([doc.material_id, doc.composition, doc.band_gap])
    print("materials_info.csv has been created successfully.")

    # Download CIF files
    cif_output_dir = "HexOx_cifs"
    os.makedirs(cif_output_dir, exist_ok=True)
    print(f"Directory '{cif_output_dir}' ensures existence for CIFs.")

    print(f"Attempting to download CIF files to '{cif_output_dir}'...")
    download_success_flag = True
    for mpid in mpids:
        try:
            # Use the working CIF download method from the manual image
            structure = mpr.materials.summary.search(material_ids=[mpid], fields=["structure"])[0].structure
            structure.to(fmt="cif", filename=os.path.join(cif_output_dir, f"{mpid}.cif"))
        except Exception as e:
            print(f"Error downloading CIF for {mpid}: {e}. Skipping this material.")
            download_success_flag = False
            continue
    print("CIF file download attempt complete.")
    if not download_success_flag:
        print(f"NOTE: Some CIF files could not be downloaded. Check '{cif_output_dir}' for completeness.")

# Generate id_prop.csv
print("Processing data for id_prop.csv...")
df = pd.read_csv('materials_info.csv')
df['Material ID'] = df['Material ID'].str.replace('mp-', '')
df_processed = df[['Material ID', 'Band Gap']]
df_processed.to_csv('id_prop.csv', header=False, index=False)
print("id_prop.csv has been created successfully.")

# Rename CIF files
directory_for_cif_rename = cif_output_dir
print(f"Renaming CIF files in '{directory_for_cif_rename}'...")

if not os.path.exists(directory_for_cif_rename) or not os.listdir(directory_for_cif_rename):
    print(f"WARNING: Directory '{directory_for_cif_rename}' is empty or does not exist. CIF renaming skipped.")
    print("Please ensure CIF files are downloaded into this directory first.")
else:
    for filename in os.listdir(directory_for_cif_rename):
        if filename.startswith('mp-') and filename.endswith('.cif'):
            new_filename = filename.replace('mp-', '')
            old_file = os.path.join(directory_for_cif_rename, filename)
            new_file = os.path.join(directory_for_cif_rename, new_filename)
            os.rename(old_file, new_file)
    print("CIF file renaming complete.")

# Placeholder for 'Hex' directory
output_dir_hex_placeholder = "Hex"
if not os.path.exists(output_dir_hex_placeholder):
    os.makedirs(output_dir_hex_placeholder)
    print(f"Directory '{output_dir_hex_placeholder}' created.")
else:
    print(f"Directory '{output_dir_hex_placeholder}' already exists.")
