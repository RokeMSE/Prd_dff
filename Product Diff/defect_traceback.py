import cv2
import numpy as np
import pandas as pd
import os
import glob

class DefectTracker:
    def __init__(self, data_file, image_dir):
        self.data_file = data_file
        self.image_dir = image_dir
        self.defect_data = self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.data_file)
        return df

    def get_historical_images(self):
        """Finds all `_In.jpg` and `_Out.jpg` images in the directory and sorts them dynamically."""
        # Find all process images
        pattern = os.path.join(self.image_dir, '*_[In|Out]*.jpg')
        # also match .png if any
        images = []
        for ext in ('*.jpg', '*.png'):
            images.extend(glob.glob(os.path.join(self.image_dir, '*_[I|O]*' + ext[1:])))

        # Parse operation number and In/Out status
        parsed_images = []
        for img_path in images:
            basename = os.path.basename(img_path)
            # Assuming format like "6275_In.jpg"
            name_parts = os.path.splitext(basename)[0].split('_')
            
            if len(name_parts) >= 2:
                try:
                    op_num = int(name_parts[0])
                    status = name_parts[1].lower() # 'in' or 'out'
                    # Out comes before In for the same operation when tracing backwards
                    status_weight = 1 if status == 'out' else 0 
                    parsed_images.append((op_num, status_weight, img_path))
                except ValueError:
                    continue
        
        # Sort descending by operation number, then by Out(1) before In(0)
        parsed_images.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        return [img_path for _, _, img_path in parsed_images]

    def preprocess_image(self, img):
        """Applies CLAHE to normalize lighting."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray)

    def align_images(self, baseline_img, target_img):
        """Aligns target_img to baseline_img using SIFT/ORB and Homography, optimized with downscaling."""
        scale = 0.25
        base_small = cv2.resize(baseline_img, (0,0), fx=scale, fy=scale)
        target_small = cv2.resize(target_img, (0,0), fx=scale, fy=scale)
        
        # Preprocess
        base_gray = self.preprocess_image(base_small)
        target_gray = self.preprocess_image(target_small)

        orb = cv2.ORB_create(nfeatures=5000)
        keypoints1, descriptors1 = orb.detectAndCompute(target_gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(base_gray, None)

        if descriptors1 is None or descriptors2 is None:
            return None, None

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep top matches
        GOOD_MATCH_PERCENT = 0.15
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        if len(points1) < 4:
            return None, None

        # Find homography
        H_small, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        
        if H_small is None:
            return None, None

        # Scale Homography back up to full size
        S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        S_inv = np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]])
        h_matrix = S_inv @ H_small @ S

        # Use homography
        height, width = baseline_img.shape[:2]
        aligned_img = cv2.warpPerspective(target_img, h_matrix, (width, height))

        return aligned_img, h_matrix

    def detect_defect(self, baseline_img, aligned_target, bbox):
        """Compares the ROI in both images to detect differences."""
        x, y, w, h = bbox
        
        # Ensure bbox is within image bounds
        bh, bw = baseline_img.shape[:2]
        x = max(0, min(x, bw - 1))
        y = max(0, min(y, bh - 1))
        w = max(1, min(w, bw - x))
        h = max(1, min(h, bh - y))

        base_roi = baseline_img[y:y+h, x:x+w]
        target_roi = aligned_target[y:y+h, x:x+w]

        base_gray = self.preprocess_image(base_roi)
        target_gray = self.preprocess_image(target_roi)
        
        # Simple absolute difference with thresholding
        diff = cv2.absdiff(base_gray, target_gray)
        _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of different pixels
        diff_score = np.sum(thresh == 255) / (w * h)
        
        # High diff score means the target image is DIFFERENT from the defective baseline.
        # So a high difference means the target image is CLEAN.
        # A low difference (< 0.05) means the target image is STILL DEFECTIVE.
        return diff_score <= 0.05, diff_score # threshold config

    def process_defect(self, row_index):
        if row_index >= len(self.defect_data):
            print("Row index out of bounds")
            return
            
        row = self.defect_data.iloc[row_index]
        print(f"--- Processing Defect {row['DR_SUB_ITEM']} ({row_index+1}/{len(self.defect_data)}) ---")
        
        # DVI baseline crop coordinates in DVI coords
        w_dvi, h_dvi = int(row['BOX_SIDE_X']), int(row['BOX_SIDE_Y'])
        x_dvi, y_dvi = int(row['BOX_CTR_X'] - w_dvi/2), int(row['BOX_CTR_Y'] - h_dvi/2)
        
        # Find the DVI base image in the directory based on the CSV path
        basename = os.path.basename(row['IMAGE_FULL_PATH'])
        dvi_image_path = os.path.join(self.image_dir, basename)
        
        hist_images = self.get_historical_images()
        if not hist_images:
            print("No historical images found in directory")
            return
            
        if os.path.exists(dvi_image_path):
            baseline_path = dvi_image_path
            print(f"Loading actual DVI baseline (OG image) from {baseline_path}")
            baseline_img = cv2.imread(baseline_path) 
            bbox = (x_dvi, y_dvi, w_dvi, h_dvi)
        else:
            # Fallback to the latest operation image and compute bounding box from M_POSITION
            baseline_path = hist_images[0]
            print(f"Loading fallback baseline from {baseline_path}")
            baseline_img = cv2.imread(baseline_path)
            
            if baseline_img is not None:
                bh, bw = baseline_img.shape[:2]
                xc, yc = bw/2.0, bh/2.0
                
                # Approximate scale: 15 um per pixel. DVI size was at 10 um per pixel.
                scale = 1/15.0
                w_um = w_dvi * 10.0
                h_um = h_dvi * 10.0
                
                pixel_w = int(w_um * scale)
                pixel_h = int(h_um * scale)
                
                # Map using M_POSITION
                # Image X goes right (positive), M_POSITION_X goes right (positive) -> ADD
                # Image Y goes down (positive), M_POSITION_Y goes up (positive) -> SUBTRACT
                center_x = xc + row['M_POSITION_X'] * scale
                center_y = yc - row['M_POSITION_Y'] * scale 
                
                x = int(center_x - pixel_w/2)
                y = int(center_y - pixel_h/2)
                bbox = (x, y, pixel_w, pixel_h)
            else:
                bbox = (0,0,0,0)
        
        if baseline_img is None:
            print("Could not load baseline image.")
            return

        report_images = []
        
        # First image in report is the baseline
        report_base = baseline_img.copy()
        x, y, w, h = bbox
        
        # Draw highly visible locator for the baseline
        cv2.rectangle(report_base, (x, y), (x+w, y+h), (0, 0, 255), 8)
        cx, cy = x + w//2, y + h//2
        cv2.circle(report_base, (cx, cy), max(200, max(w,h)*2), (0, 0, 255), 15) # Big locator circle
        cv2.putText(report_base, "OG Image", (x, max(y-200, 100)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)
        
        report_images.append(("OG Image", report_base))
        print("Tracing backwards...")
        
        for idx, hist_path in enumerate(hist_images):
            print(f"Comparing with: {os.path.basename(hist_path)}")
            target_img = cv2.imread(hist_path)
            if target_img is None:
                continue
                
            aligned_target, H_matrix = self.align_images(baseline_img, target_img)
            
            if aligned_target is None or H_matrix is None:
                print(f"  Alignment failed for {os.path.basename(hist_path)}")
                continue
            is_defective, score = self.detect_defect(baseline_img, aligned_target, bbox)
            
            
            # Map the bbox to the Native target_img to draw it without having distorted images
            try:
                # H_matrix transforms target_img -> baseline_img. So inverse(H_matrix) goes baseline -> target
                inv_H = np.linalg.inv(H_matrix)
                corners = np.array([
                    [x, y],
                    [x+w, y],
                    [x+w, y+h],
                    [x, y+h]
                ], dtype=np.float32).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(corners, inv_H)
                
                target_disp = target_img.copy()
                
                if is_defective:
                    print(f"  -> Defect detected (Score: {score:.3f}). Origin might be here or earlier.")
                    color = (0, 0, 255) # Red
                    label = f"Defect Found ({score:.2f})"
                else:
                    print(f"  -> Defect NOT detected (Score: {score:.3f}). Source is likely later in the process.")
                    color = (0, 255, 0) # Green
                    label = "Clean"
                    
                # Draw native polygon bounding box
                cv2.polylines(target_disp, [np.int32(transformed_corners)], True, color, 8)
                
                # Draw highly visible locator circle
                t_pts = transformed_corners.reshape(-1, 2)
                cx = int(np.mean(t_pts[:, 0]))
                cy = int(np.mean(t_pts[:, 1]))
                radius = max(200, int(max(w, h)) * 2)
                cv2.circle(target_disp, (cx, cy), radius, color, 15)
                
                # Put text near the locator circle
                cv2.putText(target_disp, label, (cx - radius, max(cy - radius - 20, 50)), cv2.FONT_HERSHEY_SIMPLEX, 4, color, 8)

                
                report_images.append((os.path.basename(hist_path), target_disp))
            except Exception as e:
                print("Failed to inverse project bounding box", e)
            
            if not is_defective:
                # If it's clean here, the defect started AFTER this process.
                print(f"*** Defect originated between {os.path.basename(hist_path)} and the subsequent process. ***")
                # We do not break here so we can plot the box on all tested images to validate
                
        self.generate_report(row, report_images)

    def generate_report(self, row, images_data):
        print("Generating report...")
        if not images_data:
            return
            
        # Calculate grid size (2 columns, N/2 rows)
        n_images = len(images_data)
        cols = 2
        rows = (n_images + 1) // 2
        
        # Make all images the same size for concatenation (use baseline size)
        target_h, target_w = images_data[0][1].shape[:2]
        
        # Create a blank grid
        grid = np.zeros((rows * target_h, cols * target_w, 3), dtype=np.uint8)
        
        for idx, (name, img) in enumerate(images_data):
            r = idx // cols
            c = idx % cols
            
            # Resize if necessary while preserving aspect ratio and padding
            if img.shape[:2] != (target_h, target_w):
                h, w = img.shape[:2]
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_scaled = cv2.resize(img, (new_w, new_h))
                
                img_resized = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                y_pad = (target_h - new_h) // 2
                x_pad = (target_w - new_w) // 2
                img_resized[y_pad:y_pad+new_h, x_pad:x_pad+new_w] = img_scaled
            else:
                img_resized = img
                
            y_offset = r * target_h
            x_offset = c * target_w
            
            grid[y_offset:y_offset+target_h, x_offset:x_offset+target_w] = img_resized
            
            # Add a clear label box
            cv2.rectangle(grid, (x_offset, y_offset), (x_offset+400, y_offset+60), (0,0,0), -1)
            cv2.putText(grid, name, (x_offset+10, y_offset+40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

        out_name = f"composite_report_{row['VISUAL_ID']}_{row['DR_SUB_ITEM']}.jpg"
        out_path = os.path.join(self.image_dir, out_name)
        
        # The grid can be massive, let's heavily downscale for the final output so it's viewable
        scale_percent = 100 # percent of original size
        width = int(grid.shape[1] * scale_percent / 100)
        height = int(grid.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_grid = cv2.resize(grid, dim, interpolation = cv2.INTER_AREA)

        cv2.imwrite(out_path, resized_grid)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    data_csv = r"c:\Users\tranyenq\Downloads\PRJS\Product Diff\data\DVI_box_data.csv"
    img_dir = r"c:\Users\tranyenq\Downloads\PRJS\Product Diff\data"
    
    tracker = DefectTracker(data_csv, img_dir)
    print("Files found:", tracker.get_historical_images())
    
    # Process all defects
    for i in range(len(tracker.defect_data)):
        tracker.process_defect(i)
