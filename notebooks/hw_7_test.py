import os
import pytest
import pandas as pd
from click.testing import CliRunner
from main import*

@pytest.fixture
def dataset_path():
    return "archive (4)/singapore_airlines_reviews.csv"

@pytest.fixture
def model_path():
    return "test_model.pkl"

def test_train(dataset_path, model_path):
    runner = CliRunner()
    result = runner.invoke(train, ['--data', dataset_path, '--model', model_path, '--split', '0.2'])    
    assert result.exit_code == 0
    assert os.path.exists(model_path)
    os.remove(model_path)

def test_predict_csv_input(dataset_path, model_path):
    runner = CliRunner()
    result = runner.invoke(train, ['--data', dataset_path, '--model', model_path, '--split', '0.2'])
    assert result.exit_code == 0
    result = runner.invoke(predict, ['--model', model_path, '--data', dataset_path])
    assert result.exit_code == 0
    assert result.output.count('\n') == pd.read_csv(dataset_path).shape[0]
    os.remove(model_path)

def test_predict_string_input(dataset_path, model_path):
    runner = CliRunner()
    result = runner.invoke(train, ['--data', dataset_path, '--model', model_path, '--split', '0.2'])
    assert result.exit_code == 0
    input_text = "The flight was good, but the food was terrible"
    result = runner.invoke(predict, ['--model', model_path, '--data', input_text])
    assert result.exit_code == 0
    assert int(result.output.strip()[1]) in [1, 2, 3, 4, 5]
    os.remove(model_path)

def test_data_split(dataset_path):
    df = pd.read_csv(dataset_path)
    train_size = 0.8
    random_state = 42
    train_data, test_data = train_test_split(df, train_size=train_size, random_state=random_state)
    assert len(train_data) == int(len(df) * train_size)
    assert len(test_data) == len(df) - int(len(df) * train_size)
    assert not train_data.equals(test_data)
    

if __name__ == "__main__":
    pytest.main()