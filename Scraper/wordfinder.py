import random
import csv
import string

def generate_combinations():
    characters = string.ascii_lowercase + string.digits  # 'abcdefghijklmnopqrstuvwxyz0123456789'
    
    with open('combinations.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Iterate over lengths from 1 to 40
        for length in range(6, 11):
            # Iterate over all possible combinations of the given length
            generate_for_length(writer, characters, length)

def generate_for_length(writer, characters, length):
    if length == 1:
        for char in characters:
            writer.writerow([char])
    else:
        generate_recursive(writer, characters, "", length)

def generate_recursive(writer, characters, current_str, length):
    if len(current_str) == length:
        writer.writerow([current_str])
    else:
        for char in characters:
            generate_recursive(writer, characters, current_str + char, length)

if __name__ == '__main__':
    generate_combinations()
