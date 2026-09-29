import os
import pandas as pd
import numpy as np
from pathlib import Path
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GRANULOSA CELL ROI RATIO CALCULATION SCRIPT
# ============================================================================
# This script:
# 1. Scans all subfolders for Excel files
# 2. For each subfolder: Creates [SubfolderName]_processed.xlsx with ratios
# 3. Creates Final_Ratios_Summary.xlsx in main folder with only ratio columns
# ============================================================================

class ROIRatioProcessor:
    """Process ROI data from Excel files and calculate signal-to-background ratios"""
    
    def __init__(self, main_folder="."):
        self.main_folder = Path(main_folder)
        self.all_ratios = {}  # Dictionary to store all ratios for final summary
        self.ratio_lengths = {}  # Track number of ratios per file
        
        print("=" * 80)
        print("ROI RATIO CALCULATION SCRIPT")
        print("=" * 80)
        print(f"Main folder: {self.main_folder.absolute()}\n")
    
    def calculate_ratio_for_sheet(self, df, sheet_name):
        """
        Calculate signal-to-background ratios for a single sheet
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with ROI_ID and ROI Mean Intensity columns
        sheet_name : str
            Name of the sheet (for logging)
        
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame with added "ROI Ratio" column
        """
        
        # Convert column names to strings and standardize them
        df.columns = df.columns.astype(str)
        df.columns = df.columns.str.strip()
        
        # Find ROI ID and Mean Intensity columns
        roi_id_col = None
        intensity_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'roi' in col_lower and 'id' in col_lower:
                roi_id_col = col
            if 'mean' in col_lower and 'intensity' in col_lower:
                intensity_col = col
        
        if roi_id_col is None or intensity_col is None:
            print(f"  ⚠ WARNING: Could not find ROI ID or Mean Intensity columns in '{sheet_name}'")
            print(f"    Available columns: {list(df.columns)}")
            return df
        
        # Initialize ratio column with NaN
        df['ROI Ratio'] = np.nan
        
        # Calculate ratios for odd ROI IDs (cells)
        for idx, row in df.iterrows():
            roi_id = row[roi_id_col]
            
            # Only calculate ratio for odd ROI IDs (granulosa cells)
            if pd.notna(roi_id) and roi_id % 2 == 1:
                cell_intensity = row[intensity_col]
                
                # Find corresponding background (next even ROI ID)
                bg_roi_id = roi_id + 1
                bg_row = df[df[roi_id_col] == bg_roi_id]
                
                if len(bg_row) > 0:
                    bg_intensity = bg_row[intensity_col].values[0]
                    
                    # Calculate ratio (handle division by zero)
                    if pd.notna(bg_intensity) and bg_intensity > 0:
                        ratio = cell_intensity / bg_intensity
                        df.at[idx, 'ROI Ratio'] = ratio
                    else:
                        df.at[idx, 'ROI Ratio'] = np.nan
        
        return df
    
    def process_excel_file(self, file_path):
        """
        Process a single Excel file (handle multiple sheets if present)
        
        Parameters:
        -----------
        file_path : Path
            Path to the Excel file
        
        Returns:
        --------
        processed_sheets : dict
            Dictionary with sheet names as keys and processed DataFrames as values
        """
        
        processed_sheets = {}
        file_name = file_path.stem  # Filename without extension
        
        try:
            # Read all sheets from Excel file
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            # Check if multiple sheets exist
            if len(sheet_names) > 1:
                print(f"  📄 Found {len(sheet_names)} sheets in '{file_path.name}'")
            
            # Process each sheet
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Skip empty sheets
                if df.empty or len(df.columns) == 0:
                    print(f"    ⏭️  Skipping empty sheet: '{sheet_name}'")
                    continue
                
                # Check if this sheet has the ROI columns
                col_names = [str(col).lower() for col in df.columns]
                has_roi_id = any('roi' in col and 'id' in col for col in col_names)
                has_intensity = any('mean' in col and 'intensity' in col for col in col_names)
                
                if not (has_roi_id and has_intensity):
                    print(f"    ⏭️  Skipping sheet '{sheet_name}' (no ROI columns found)")
                    continue
                
                # Process the sheet
                processed_df = self.calculate_ratio_for_sheet(df, sheet_name)
                processed_sheets[sheet_name] = processed_df
                
                # Extract only the ratio column for final summary
                if 'ROI Ratio' in processed_df.columns:
                    ratios = processed_df['ROI Ratio'].dropna().reset_index(drop=True)
                    
                    if file_name not in self.all_ratios:
                        self.all_ratios[file_name] = ratios
                        self.ratio_lengths[file_name] = len(ratios)
                    
                    num_ratios = len(ratios)
                    print(f"    ✓ Processed: {sheet_name} ({len(processed_df)} rows, {num_ratios} ratios)")
        
        except Exception as e:
            print(f"  ❌ ERROR processing '{file_path.name}': {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
        
        return processed_sheets
    
    def save_processed_file_for_folder(self, subfolder_path):
        """
        Process all Excel files in a subfolder and save as single Excel file with multiple sheets
        
        Parameters:
        -----------
        subfolder_path : Path
            Path to the subfolder
        """
        
        subfolder_name = subfolder_path.name
        print(f"\n{'='*80}")
        print(f"PROCESSING SUBFOLDER: {subfolder_name}")
        print(f"{'='*80}")
        
        # Find all Excel files in this subfolder
        excel_files = list(subfolder_path.glob("*.xlsx")) + list(subfolder_path.glob("*.xls"))
        
        # Filter out already processed files
        excel_files = [f for f in excel_files if not f.name.endswith("_processed.xlsx")]
        
        if len(excel_files) == 0:
            print(f"  ⚠ No Excel files found in '{subfolder_name}'")
            return
        
        print(f"  Found {len(excel_files)} Excel file(s)")
        
        # Process each file and store all sheets
        all_processed_sheets = {}
        
        for excel_file in excel_files:
            file_name = excel_file.stem
            print(f"\n  Processing: {excel_file.name}")
            
            processed_sheets = self.process_excel_file(excel_file)
            
            for sheet_name, df in processed_sheets.items():
                # Use filename as sheet name in output (or append sheet name if multiple sheets in source)
                if len(processed_sheets) == 1:
                    output_sheet_name = file_name
                else:
                    output_sheet_name = f"{file_name}_{sheet_name}"
                
                # Keep sheet name under Excel limit (31 characters)
                if len(output_sheet_name) > 31:
                    output_sheet_name = output_sheet_name[:28] + "..."
                
                all_processed_sheets[output_sheet_name] = df
                print(f"    ✓ Processed: {excel_file.name} (Rows: {len(df)}, Ratios: {(~df['ROI Ratio'].isna()).sum()})")
        
        # Save all processed sheets to output file
        if all_processed_sheets:
            output_file = subfolder_path / f"{subfolder_name}_processed.xlsx"
            
            try:
                # Create Excel writer
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    for sheet_name, df in all_processed_sheets.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"\n  ✓ SAVED: {output_file.name}")
                print(f"    Location: {output_file}")
                print(f"    Total sheets: {len(all_processed_sheets)}")
            except Exception as e:
                print(f"\n  ❌ ERROR saving {output_file.name}: {str(e)}")
        else:
            print(f"\n  ⚠ No valid data sheets found to save in {subfolder_name}")
    
    def create_final_summary(self):
        """
        Create final summary file with only ratio columns
        Output: Final_Ratios_Summary.xlsx in main folder
        """
        
        if not self.all_ratios:
            print("\n⚠ No ratios found. Cannot create summary file.")
            return
        
        print(f"\n{'='*80}")
        print("CREATING FINAL SUMMARY FILE")
        print(f"{'='*80}")
        
        # Find maximum number of ratios across all files
        max_ratios = max(self.ratio_lengths.values())
        
        # Create DataFrame for summary
        summary_df = pd.DataFrame()
        
        for file_name in sorted(self.all_ratios.keys()):
            ratios = self.all_ratios[file_name]
            
            # Pad with NaN to match max length
            padded_ratios = pd.Series(
                list(ratios) + [np.nan] * (max_ratios - len(ratios))
            )
            
            summary_df[file_name] = padded_ratios
            print(f"  ✓ Added column: {file_name} ({len(ratios)} ratios)")
        
        # Save summary file
        output_file = self.main_folder / "Final_Ratios_Summary.xlsx"
        summary_df.to_excel(output_file, sheet_name="Ratios", index=False)
        
        print(f"\n  ✓ SAVED: {output_file.name}")
        print(f"    Location: {output_file}")
        print(f"    Dimensions: {summary_df.shape[0]} rows × {summary_df.shape[1]} columns")
    
    def run(self):
        """
        Main execution function - process all subfolders and create summary
        """
        
        # Find all subfolders
        subfolders = [d for d in self.main_folder.iterdir() 
                     if d.is_dir() and not d.name.startswith('.')]
        
        if not subfolders:
            print("⚠ No subfolders found in main directory")
            return
        
        print(f"Found {len(subfolders)} subfolder(s)\n")
        
        # Process each subfolder
        for subfolder in sorted(subfolders):
            self.save_processed_file_for_folder(subfolder)
        
        # Create final summary
        self.create_final_summary()
        
        # Print completion summary
        print(f"\n{'='*80}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*80}")
        print(f"\nTotal files processed: {len(self.all_ratios)}")
        print(f"Total unique age groups: {len([d for d in self.main_folder.iterdir() if d.is_dir()])}")
        print(f"\nGenerated files:")
        print(f"  1. [SubfolderName]_processed.xlsx in each subfolder")
        print(f"  2. Final_Ratios_Summary.xlsx in main folder")
        print(f"\n✓ All operations completed successfully!\n")


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize processor
    processor = ROIRatioProcessor(main_folder=".")
    
    # Run the processing pipeline
    processor.run()
    
    print("\n" + "="*80)
    print("Script finished. Check the output files in your directories.")
    print("="*80 + "\n")