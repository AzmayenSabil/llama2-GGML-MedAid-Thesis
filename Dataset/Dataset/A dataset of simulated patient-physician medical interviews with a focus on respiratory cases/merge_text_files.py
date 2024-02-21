import os

# Directory containing your text files
directory = './text'

# Initialize an empty list to store conversations
conversations = []

# Iterate through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):  # Check if it's a text file
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            conversation = file.read().strip()  # Read the entire file as a single conversation
            conversations.append(conversation)

# Save the combined conversations with scenario headers to a new file
output_file = 'combined_conversations_with_scenarios.txt'
with open(output_file, 'w') as file:
    current_scenario = 1  # Initialize scenario counter
    for conv in conversations:
        # Add a scenario header at the start of each scenario
        file.write(f"[Scenario {current_scenario}]\n\n")
        file.write(conv + '\n\n[SEP]\n\n')  # Separate conversations with [SEP]
        current_scenario += 1  # Increment scenario counter
