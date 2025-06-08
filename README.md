# Covid-19-Detection-using-Federated-Learning
The project aims to revolutionize COVID-19 diagnosis by harnessing the power of Federated  Learning, an innovative approach that enables collaborative model development without sharing raw patient data.
Federated learning empowered Covid-19 detection using Chest X-Ray images

**Developing an efficient federated learning system for COVID-19 detection from Chest X-Ray images to address privacy concerns and Non-IID data**.
**Experiments**
**Experiment 1:**
Equal Data Division among Clients
To investigate the performance of a federated learning model when the data is divided equally among the participating clients. We divided the X-ray images of COVID-19 patients equally among the four clients participating in the federated learning setup. 
**Experiment 2:**
Unequal Data Division Based on Percentages.
To assess the impact of dividing the data among clients based on varying percentages. X-ray images of COVID-19 patients were divided among the four clients based on predetermined percentages.
**Experiment 3:**
Data Division Using Ratios Creating Imbalance
To explore the effects of creating data imbalance among clients by dividing the data based on specific ratios. Data division was conducted with an intentional imbalance among the clients, simulating real-world scenarios where certain clients may have more data than others. The performance of the federated learning model was evaluated under imbalanced data conditions.
**Experiment 4:**
Mixed Data Types among Clients
To examine the performance of a federated learning model when clients possess different types of medical imaging data. X-ray images of COVID-19 patients were distributed among three clients, while one client received CT scan images, representing a heterogeneous data setup. 
Directory Structure
client.py: Client-side script for federated learning.
server.py: Server-side script for federated learning.
script.py, script2.py, script3.py, script4.py : 4 files for distributing data for the 4 experiments respectively.
exp1, exp2, exp3, exp4: 4 folders for storing log files of each experiment.

**Setup and Usage**
**Prerequisites**
•	Python 3.x
•	Required Python packages including TensorFlow and Flower (flwr)
•	Install TensorFlow and Flower:
Pip install tensorflow and flwr
**Running Data Distribution Scripts**
**Run the four scripts to distribute data:**
•	python distribute_data_1.py
•	python distribute_data_2.py
•	python distribute_data_3.py
•	python distribute_data_4.py

**Ensure that the log directories for each experiment are correctly set in both client.py and server.py. For each experiment, you will have to create a corresponding log directory**:
•	exp1
•	exp2
•	exp3
•	exp4

**Update the dataset paths for training data in the client.py file as follows:**
if __name__ == '__main__':
    if len(sys.argv) > 1:
        client_n = int(sys.argv[1])
        dataset = get_dataset('Dataset1/client' + str(client_n))

**Update the dataset paths for testing data in the client.py file as follows**:
test_ds = tf.keras.utils.image_dataset_from_directory("Dataset1/Test", seed=123, image_size=(224, 224) 
**Running the Federated Learning Experiments**
**To start an experiment, run the following commands**:
**Start the server:**
Open the command prompt in the project file and run the server as follows:
Python server.py
**Start 4 clients :**
Open the 4 windows of command prompt in the project file and run the 4 clients in each as follows:
•	Python client.py 1
•	Python client.py 2
•	Python client.py 3
•	Python client.py 4
**Repeat these steps for each experiment, ensuring you update the log directory path and dataset path before each run.**
**Notes**
Ensure that the dataset paths in client.py are correctly set before running the experiments.
Modify the log directory path in both client.py and server.py for each experiment to avoid log file conflicts.

