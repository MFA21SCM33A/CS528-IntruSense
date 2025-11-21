import pandas as pd
import os


def create_nsl_kdd():
    INPUT_DIR= "data/raw/NSL_KDD"
    OUTPUT_FILE="data/raw/NSL_KDD_raw.csv"
    files = [
        "KDDTrain+.txt",
        "KDDTest+.txt"
    ]
    columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty_level"
    ]

    dfs = []
    print("\nSTART -----------[NSL KDD]-----------")
    for f in files:
        path = os.path.join(INPUT_DIR, f)
        #print(f"Read & Merge : {path}")
        df = pd.read_csv(path, header=None)
        dfs.append(df)
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.columns=columns
    print("Records shape:", final_df.shape)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"File Created: {OUTPUT_FILE}")
    print("END -----------[NSL KDD]-----------\n")

def create_cicids_2017():
    INPUT_DIR= "data/raw/CICIDS2017"
    OUTPUT_FILE= "data/raw/CICIDS2017_raw.csv"
    print("START -----------[CICIDS2017]-----------") 
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    print(f"Total File:  {len(files)}")

    dfs = []
    for file in files:
        #print(f"Read & Merge : {file}")
        df = pd.read_csv(os.path.join(INPUT_DIR, file))
        #df["SourceFile"] = file
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    print("Records shape:",final_df.shape)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"File Created: {OUTPUT_FILE}")
    print("END -----------[CICIDS2017]-----------\n")

def main():
    create_nsl_kdd()
    create_cicids_2017()

if __name__ == "__main__":
    main()