import os

def gen_report(options: dict, string: str):
  report_string = """
# System Performance Report

## Data Analysis

Tests indicate the following about the data in the file `""" + options["filename"] + """`:
  * """ + options["normality_message"] + """
  * """ + options["skewness_message"] + """
  * """ + options["test_statistic_message"] + """


Based on this information, the following significance tests are appropriate for your data:

""" + options["significance_tests_table"] + """

## Significance testing

Requiring a significance level $\\alpha = """ + options["significance_alpha"] + """$, """ + options["bootstrap iterations"] + """ iterations for bootstraping tests, and an expected mean difference for null hypothesis mean of """ + options["expected_mean_diff"] + """ and using the """ + options["chosen_sig_test"] + """ significance test, we can conclude that you """ + options["should_reject?"] + """ the null hypothesis. The test statistic and confidence interval are """ + options["statistic/CI"] + """ respectively.

## Effect Size

## Power Analysis


"""

  os.system("echo "" > user/report.md")
  for line in report_string.split("\n"):
    print(line, file=open('user/report.md', 'a'))
  os.system("zip -r user/report.zip user/*")
