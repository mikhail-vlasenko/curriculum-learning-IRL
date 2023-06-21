import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from config import CONFIG, get_demo_name
from envs.env_factory import make_env


def process_trajectories(trajectories):
    # Iterate over each trajectory and extract state-action pairs
    state_action_pairs = []
    for episode in trajectories:
        states = episode['states']
        actions = episode['actions']
        for state, action in zip(states, actions):
            # Append state-action pair to the list
            # Here, we are assuming that 'state' and 'action' are simple numeric arrays.
            # If they are not, you will need to convert them to a suitable format.
            state_action_pairs.append(list(state) + [action])
    return state_action_pairs


device = CONFIG.device
expert_trajectories = pickle.load(open(get_demo_name(), 'rb'))
airl_trajectories = pickle.load(open('demonstrations/from_airl_policy_neg_1_rew.pk', 'rb'))
# airl_trajectories = pickle.load(open('demonstrations/ppo_demos_size-10_tile-reward_reward-conf-default_old.pk', 'rb'))

expert_trajectories = process_trajectories(expert_trajectories)
airl_trajectories = process_trajectories(airl_trajectories)

# Create environment
env = make_env()
obs_shape = env.observation_space.shape

# Define the model
discriminator = nn.Sequential(
    nn.Linear(obs_shape[0] + 1, CONFIG.discriminator.dimensions[0]),
    nn.ReLU(),
    nn.Linear(CONFIG.discriminator.dimensions[0], CONFIG.discriminator.dimensions[1]),
    nn.ReLU(),
    nn.Linear(CONFIG.discriminator.dimensions[1], 1),
    nn.Sigmoid(),
).to(device)

# Convert trajectories to tensors and create labels
expert_trajectories_tensor = torch.FloatTensor(expert_trajectories)
airl_trajectories_tensor = torch.FloatTensor(airl_trajectories)

expert_labels = torch.ones((expert_trajectories_tensor.size(0), 1))
airl_labels = torch.zeros((airl_trajectories_tensor.size(0), 1))

# Concatenate data and labels
data = torch.cat((expert_trajectories_tensor, airl_trajectories_tensor))
labels = torch.cat((expert_labels, airl_labels))

# Create a dataset and split into training and testing
dataset = TensorDataset(data, labels)
train_ratio = 0.8
n_train_examples = int(len(dataset) * train_ratio)
n_test_examples = len(dataset) - n_train_examples
train_dataset, test_dataset = random_split(dataset, [n_train_examples, n_test_examples])

# Create data loaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# Training loop
for epoch in tqdm(range(30)):  # Adjust the number of epochs
    for i, (inputs, labels) in enumerate(train_dataloader):
        # Clear gradients
        optimizer.zero_grad()

        # Run inputs through model
        outputs = discriminator(inputs.to(device))

        # Compute loss
        loss = criterion(outputs.cpu(), labels)

        # Backpropagate gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

    # Print loss every epoch
    if epoch % 10 == 9:
        print(f'Epoch {epoch}/{100}, Loss: {loss.item()}')

# Testing the model
discriminator.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:
        # Forward pass
        outputs = discriminator(inputs.to(device)).cpu()

        # Get the predicted class
        predicted = (outputs > 0.5).float()

        # Update total and correct counts
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

# Compute accuracy
accuracy = correct_predictions / total_predictions
print(f'Test Accuracy: {accuracy * 100:.2f}%')

correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for inputs, labels in train_dataloader:
        # Forward pass
        outputs = discriminator(inputs.to(device)).cpu()

        # Get the predicted class
        predicted = (outputs > 0.5).float()

        # Update total and correct counts
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

# Compute accuracy
accuracy = correct_predictions / total_predictions
print(f'Train Accuracy: {accuracy * 100:.2f}%')
