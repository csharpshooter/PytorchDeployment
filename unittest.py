import os
import csv


class TestClass:

    def test_check_if_model_file_present_in_root_folder(self):
        assert os.path.isfile('./train/model.pth') == False

    def test_check_if_data_folder_present_in_root_folder(self):
        assert os.path.isdir('./train/data') == False

    def test_check_if_metrics_csv_present_in_root_folder(self):
        assert os.path.isfile('./train/metrics.csv') == True

    def test_validate_train_accuracy_greater_than_70_pct(self):
        with open("./train/metrics.csv", 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                if row.__len__() > 0 and row[0].__contains__('Train'):
                    assert (float(row[0].split(':')[1]) > 70) == True

    def test_validate_test_accuracy_greater_than_70_pct(self):
        with open("./train/metrics.csv", 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                if row.__len__() > 0 and row[0].__contains__('Test'):
                    assert (float(row[0].split(':')[1]) > 70) == True

    def test_validate_individual_class_accuracy_greater_than_70_pct(self):
        with open("./train/metrics.csv", 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                if row.__len__() > 0 and row[0].__contains__('plane'):
                    assert (float(row[1]) > 70) == True
                if row.__len__() > 0 and row[0].__contains__('car'):
                    assert (float(row[1]) > 70) == True
                if row.__len__() > 0 and row[0].__contains__('bird'):
                    assert (float(row[1]) > 70) == True
                if row.__len__() > 0 and row[0].__contains__('cat'):
                    assert (float(row[1]) > 70) == True
                if row.__len__() > 0 and row[0].__contains__('deer'):
                    assert (float(row[1]) > 70) == True
                if row.__len__() > 0 and row[0].__contains__('frog'):
                    assert (float(row[1]) > 70) == True
                if row.__len__() > 0 and row[0].__contains__('dog'):
                    assert (float(row[1]) > 70) == True
                if row.__len__() > 0 and row[0].__contains__('horse'):
                    assert (float(row[1]) > 70) == True
                if row.__len__() > 0 and row[0].__contains__('ship'):
                    assert (float(row[1]) > 70) == True
                if row.__len__() > 0 and row[0].__contains__('truck'):
                    assert (float(row[1]) > 70) == True
