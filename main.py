from DataLoader import DataLoader
def main():
    data_loader = DataLoader("Natural-Questions-Filtered.csv")
    data_loader.prepare_filtered_data("output.csv")

if __name__ == "__main__":
    main()