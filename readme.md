# NLPStatTest

## About

This website is a demonstration of "NLPStatTest: A Toolkit for Comparing NLP System Performance", by Haotian Zhu, Denise Mak, Jesse Gioannini, and Fei Xia. Please see the `doc` directory for the manual, and the demonstration paper, which contains a list of references.

While statistical significance testing has been commonly used to compare NLP system performance, a small *p*-value alone is not sufficient because statistical significance is different from practical significance. To measure practical significance, we recommend estimating and reporting of effect size. It is also necessary to conduct power analysis to ensure that the test corpus is large enough to achieve a desirable power level. We propose a three-stage procedure for comparing NLP system performance, and provide a toolkit, NLPStatTest, to automate the testing stage of the procedure. For future work, we will extend this work to hypothesis testing with multiple datasets or multiple metrics.

## Setup

1. Install [Conda](https://www.anaconda.com/products/individual) on your system. Read [this article](https://pythonspeed.com/articles/conda-dependency-management/) for more information about how Conda lockfiles allow us to build a reproducible environment.

	a. (On Windows) open Anaconda Prompt from the start menu. Then use `cd` to navigate to the unzipped package directory.
	b. (On Mac or Linux) open your favorite shell with `conda` aliased to your installation. Then use `cd` to navigate to the unzipped package directory.

2. From the `env` subdirectory of the `src` directory of this package, run `conda create --name nlp-stat-test --copy --file conda-linux-64.lock`, but with `linux-64` changed to `win-64` or `osx-64` as needed. 

3. Then run `conda activate nlp-stat-test`

4. Run `python run-gui.py` from the top level of the package and open `localhost:5000` in your browser (we recommend a Chromium-based browser). Keep the all the contents of the package together; separating them will break paths. The program will create a `user` subdirectory to store temporary files, and may write error logs, so writing permissions is required for the `nlp-stat-test` directory.

5. To exit the Conda environment, run `conda deactivate`.

## Citation
```
  @inproceedings{zhu-etal-2020-nlpstattest,
      title = "{NLPS}tat{T}est: A Toolkit for Comparing {NLP} System Performance",
      author = "Zhu, Haotian  and
        Mak, Denise  and
        Gioannini, Jesse  and
        Xia, Fei",
      booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing: System Demonstrations",
      month = dec,
      year = "2020",
      address = "Suzhou, China",
      publisher = "Association for Computational Linguistics",
      url = "https://www.aclweb.org/anthology/2020.aacl-demo.7",
      pages = "40--46",
      abstract = "Statistical significance testing centered on p-values is commonly used to compare NLP system performance, but p-values alone are insufficient because statistical significance differs from practical significance. The latter can be measured by estimating effect size. In this pa-per, we propose a three-stage procedure for comparing NLP system performance and provide a toolkit, NLPStatTest, that automates the process. Users can upload NLP system evaluation scores and the toolkit will analyze these scores, run appropriate significance tests, estimate effect size, and conduct power analysis to estimate Type II error. The toolkit provides a convenient and systematic way to compare NLP system performance that goes beyond statistical significance testing.",
  }
```
