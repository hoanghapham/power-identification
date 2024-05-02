import csv
import os
import time
import sys


# ATTENTION: according to their description, 0 indicates opposition and 1 indicates coalition;
# however, for Ukrainian dataset, these labels where swapped!
# If your labels are correct, simply swap the values in the dictionary below.
# I use yes/no to make it more intuitive

yes_no_dict = {
    'yes': '0',
    'no': '1'
}


def text(rows, row_index):
    print(rows[row_index]['text'])

def label(rows, row_index):
    true_label = rows[row_index]['label']
    return true_label

def translation(rows, row_index):
    print(rows[row_index]['text_en'])

def sex(rows, row_index):
    print(rows[row_index]['sex'])

def guess_label(rows, row_index, yes_no):
    true_label = label(rows, row_index)
    print('The true label is: ', true_label)
    print(true_label == yes_no_dict[yes_no])


def smooth_reading(rows, row_index):
    while True:
        print('\n')
        text(rows, row_index)
        time.sleep(1)
        while True:
            guess = input('Enter your guess: ')
            if guess == 'exit':
                sys.exit()
            if guess not in yes_no_dict:
                continue
            break
        guess_label(rows, row_index, guess)
        exit = input(
            'Enter "exit" to exit the program; \n'
            #'enter "tr" to see the translation of the text: \n'
            #'enter "s" to see the sex of the speaker: \n'
            'Otherwise, press Enter to continue: \n'
        )
        if exit == 'exit':
            sys.exit()
        elif exit == 'tr':
            translation(rows, row_index)
        elif exit == 's':
            sex(rows, row_index)
        row_index += 1



if __name__ == '__main__':
    file_name = input('Enter the file name: ')
    if file_name == 'exit':
        sys.exit()
    if os.path.isfile(file_name):
        with open(file_name, 'r') as file:
            rows = list(csv.DictReader(file, delimiter='\t'))
    else:
        print('File not found')
        print('Make sure you are in the trainingset-ideology-power/power/ directory')
        sys.exit()
    while True:
        row_index = input(
            'Enter the row from which you wanna start reading your document or "exit" to exit: '
        )
        if row_index == 'exit':
            sys.exit()
        elif not row_index.isdigit():
            print('Please enter a valid number')
            continue
        row_index = int(row_index)
        if row_index > len(rows) - 1:
            print('You reached the last row')
            sys.exit()
        smooth_reading(rows, row_index)
            


