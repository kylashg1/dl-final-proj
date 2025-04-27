TARGET_DIR="../prot_loc_proj"

# Download tif files from OpenCell AWS S3 bucket
../aws/dist/aws s3 cp s3://czb-opencell/microscopy/raw/ "$TARGET_DIR" \
--recursive --no-sign-request --exclude "*" --include "*_proj.tif";

# Flatten directory structure
find "$TARGET_DIR" -mindepth 2 -type f -exec mv "{}" "$TARGET_DIR"/ \;
find "$TARGET_DIR" -type d -empty -delete;

# Rename each file to remove "_proj"
for file in "$TARGET_DIR"/*_proj.tif; do
    # Compute the new filename by removing '_proj' before the .tif extension.
    newfile="${file/_proj.tif/.tif}"
    # Rename the file
    mv "$file" "$newfile"
done
