import os

def truefalseToYesNo(true_or_false):
  if true_or_false == 'true' or true_or_false == 'True':
    return 'Yes'
  elif true_or_false == 'false' or true_or_false == 'False':
    return 'No'
  else:
    return 'Unknown'

def gen_report(options: dict, string: str):
  report_string = """
# System Performance Report

## Data Analysis

Tests indicate the following about the data in the file `""" + options["filename"] + """`:
  * Is the data distribution normal? """ + truefalseToYesNo(options["normality_message"])  + """
  * The skewness measurement is: """ + options["skewness_message"] + """
  * Based on skewness and normality, the recommended statistic to use for each evaluation unit is: """ + options["test_statistic_message"] + """


Based on this information, the following significance tests are appropriate for your data:

""" + options["significance_tests_table"] + """

## Significance testing

Requiring a significance level $\\alpha = """ + options["significance_alpha"] + """$, """ + options["bootstrap iterations"] + """ iterations for bootstraping tests""" +  """ and using the """ + options["chosen_sig_test"] + """ significance test, can you reject the null hypothesis? """ + truefalseToYesNo(options["should_reject?"]) + """ The test statistic or confidence interval = """ + options["statistic/CI"] + """ for the significance test you selected.

## Effect Size

## Power Analysis


"""

  os.system("echo "" > user/report.md")
  for line in report_string.split("\n"):
    print(line, file=open('user/report.md', 'a'))
  os.system("zip -r user/report.zip user/*")
