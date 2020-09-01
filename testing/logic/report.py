import os

def truefalseToYesNo(true_or_false):
  if true_or_false == 'true' or true_or_false == 'True':
    return 'Yes'
  elif true_or_false == 'false' or true_or_false == 'False':
    return 'No'
  else:
    return 'Unknown'

def getReasonsTable(dictSignificanceReasons):
  table_string = ''
  for test, reason in dictSignificanceReasons:
    table_string = table_string + '* ' +   str(test) + ": {}\n".format(reason)
  return table_string

def getSigtestBootstrap():
  sigtestBootstrap = '200'
  return sigtestBootstrap

def gen_report(options: dict, string: str):
  report_string = """
# System Performance Report

## Data Analysis

Tests indicate the following about the data in the file `""" + options["filename"] + """`:
  * Is the data distribution normal? """ + truefalseToYesNo(options["normality_message"])  + """
  * The skewness measurement is: """ + options["skewness_message"] + """
  * Based on skewness and normality, the recommended statistic to use for each evaluation unit is: """ + \
                  options["test_statistic_message"] + """


Based on this information, the following significance tests are appropriate for your data:

""" + getReasonsTable(options["significance_tests_table"]) + """

## Significance testing

###Significance test parameters:
* Significance level alpha = """ + options["significance_alpha"] +\
                  '\n * Bootstrap iterations: ' + \
                  options["bootstrap iterations"]  + '\n' + \
                  """
                  * Significance test name = """ + options["chosen_sig_test"] + '\n' + """
                  * Can you reject the null hypothesis? """ + '\n' + \
                  truefalseToYesNo(options["should_reject?"]) + """
                  * Test statistic or confidence interval = """ + \
                  options["statistic/CI"] + """

## Effect Size

## Power Analysis


"""

  os.system("echo "" > user/report.md")
  for line in report_string.split("\n"):
    print(line, file=open('user/report.md', 'a'))
  os.system("zip -r user/report.zip user/*")
