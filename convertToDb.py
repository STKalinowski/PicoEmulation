import csv
import duckdb
import json


def processKeys(oldString):
    # Convert string to list
    oldList = json.loads(oldString)
    # Create binary one-hot representation for each list
    newList = []
    for inner_list in oldList:
        # Initiate binary integer
        binary_int = 0
        for index in inner_list:
            # Shift '1' to the correct place and add to the binary integer
            binary_int += 1 << index
        newList.append(binary_int)
    return newList

def create_table(connection):
    connection.execute("CREATE TABLE IF NOT EXISTS recordings (id INTEGER, inputs INT[], videofile STRING, game STRING)")

def insert_row(connection, row):
    id = int(row[0])
    keys = row[1]
    videoFile = row[0]+'.mp4'
    game = str(row[2])
    keys = processKeys(keys)
    connection.execute(f"INSERT INTO recordings VALUES (?, ?, ?, ?)", [id, keys, videoFile, game])

def process_csv_file(file_path, connection):
    with open(file_path, "r") as file:
        reader = csv.reader(file, delimiter=",")

        # Skip the header row if present
        next(reader, None)

        for row in reader:
            insert_row(connection, row)

def main():
    file_path = "./Video/2305180056.csv"
    connection = duckdb.connect(database="recordings.db")

    create_table(connection)
    print("BEGIN")
    process_csv_file(file_path, connection)

    connection.close()

if __name__ == "__main__":
    main()
