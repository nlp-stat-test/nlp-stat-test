import sys

from logic.errorHandling import InputError


def read_score_file(score_file):
	"""
	This is a function to read the input score file. We assume the file contains two
	columns of scores, named score1 and score2, separated by whitespace.

	@param score_file: the input score file
	@return: a list of two dictionaries, first is score1, second is score2
	"""

	with open(score_file) as score:
		score_lines = score.readlines()

	score1 = {}
	score2 = {}
	ind = 0
	for i in score_lines:
		line_tokens = i.split()
		if len(line_tokens) > 0:
			if len(line_tokens) != 2:
				sys.stderr.write("Input file not in the correct format (separated by a whitespace)!")
				# also raise error
				raise InputError(ind+1, "Each line must be two values separated by whitespace.")
				return
			else:
				score1[ind] = float(line_tokens[0])
				score2[ind] = float(line_tokens[1])
				ind += 1
	return([score1,score2])


def print_eu(score1, score2, score_diff, orig_ind, sample_file_dir):
	with open(sample_file_dir+'/score','w') as f:
		f.write('score1'+' '+'score2'+' '+'score_diff'+' '+'original_ind'+'\n')
		for i in score1.keys():
			f.write(str(score1[i])+' '+str(score2[i])+' '+str(score_diff[i])+' '+str(orig_ind[i])+'\n')



