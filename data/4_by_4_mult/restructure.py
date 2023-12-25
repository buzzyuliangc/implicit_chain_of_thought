
def extract_components(text):
	split_pattern_1 = "||"
	split_pattern_2 = "####"
	if split_pattern_1 not in text:
		return None
	if split_pattern_2 not in text:
		return None
	question, cot_answer = text.strip().split(split_pattern_1, 1)
	cot, answer = cot_answer.strip().split(split_pattern_2,1)
	return question, cot, answer

def main():
	combined_lines = []
	with open('train.txt', 'r') as file:
	    lines = file.readlines()

	for i in range(0, len(lines), 2):
		if(i+1 >= len(lines)): 
			continue

		# Combine two lines, removing the newline character from the first line

		q1,c1,a1 = extract_components(lines[i])

		q2,c2,a2 = extract_components(lines[i+1])
	    
		q = q1 + ' , ' + q2
		c = c1 + ' , ' + c2
		a = a1 + ' , ' + a2
		combined_line = q + "||" + c + " #### " + a
		combined_lines.append(combined_line)

	# Write the combined lines to a new file
	with open('train_combined.txt', 'w') as file:
	    for line in combined_lines:
	        file.write(line + '\n')


if __name__ == "__main__":
    main()