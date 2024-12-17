import os
import pandas as pd
import geojson

# Define root and destination folders
root_folder = '/workspace/hnsc_for_tumor/tum_ntum/tum_ntum_new/inference/csv/inf_resnet34_new'  # Replace with the path to your root folder
destination_folder = '/workspace/hnsc_for_tumor/tum_ntum/tum_ntum_new/inference/geojson/inf_geojson_resnet34_new'  # Replace with the path to your destination folder

# Ensure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Function to create GeoJSON feature from a row of data
def create_feature(row):
    point = geojson.Point((row['dim1'], row['dim2']))
    properties = {"probability": row['probability']}
    return geojson.Feature(geometry=point, properties=properties)

# Iterate over all CSV files in the root folder
for file_name in os.listdir(root_folder):
    if file_name.endswith('.csv'):
        # Full path of the CSV file
        csv_file_path = os.path.join(root_folder, file_name)
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Create GeoJSON features from the DataFrame
        features = [create_feature(row) for _, row in df.iterrows()]
        
        # Create a GeoJSON FeatureCollection
        feature_collection = geojson.FeatureCollection(features)
        
        # Define the output GeoJSON file path
        output_file_name = os.path.splitext(file_name)[0] + '.geojson'  # Replace .csv with .geojson
        output_file_path = os.path.join(destination_folder, output_file_name)
        
        # Save to the GeoJSON file
        with open(output_file_path, 'w') as f:
            geojson.dump(feature_collection, f)
        
        print(f"GeoJSON file saved as {output_file_path}")

print("Conversion complete for all CSV files.")
