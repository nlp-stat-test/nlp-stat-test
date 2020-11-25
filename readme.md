# NLPStatTest

## About

This website is a demonstration of "NLPStatTest: A Toolkit for Comparing NLP System Performance", by Haotian Zhu, Denise Mak, Jesse Gioannini, and Fei Xia. Please see the `doc` directory for the manual, and the demonstration paper, which contains a list of references.

While statistical significance testing has been commonly used to compare NLP system performance, a small *p*-value alone is not sufficient because statistical significance is different from practical significance. To measure practical significance, we recommend estimating and reporting of effect size. It is also necessary to conduct power analysis to ensure that the test corpus is large enough to achieve a desirable power level. We propose a three-stage procedure for comparing NLP system performance, and provide a toolkit, NLPStatTest, to automate the testing stage of the procedure. For future work, we will extend this work to hypothesis testing with multiple datasets or multiple metrics.

## Setup

1. Install [Conda](https://www.anaconda.com/products/individual) on your system. Read [this article](https://pythonspeed.com/articles/conda-dependency-management/) for more information.

2. From the `env` subdirectory of the `src` directory of this package, run `conda create --name nlp-stat-test --copy --file conda-linux-64.lock`, but with `linux-64` changed to `win-64` or `osx-64` as needed. Then run `conda activate nlp-stat-test`

3. Run `python run-gui.py` and open `localhost:5000` in your browser (we recommend a Chromium-based browser). Keep the all the contents of the package together; separating them will break paths. The program will create a `user` subdirectory to store temporary files, and may write error logs, so writing permissions is required for the `nlp-stat-test` directory.

4. To exit the Conda environment, run `conda deactivate`.

## Citation
```
@inproceedings{zhu-etal-2020-AACL,
  title = "{NLPStatTest: A Toolkit for Comparing NLP System Performance}",
  author = "Haotian Zhu and Denise Mak and Jesse Gioannini and Fei Xia",
  booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing (AACL-IJCNLP 2020)",
  publisher = "Association for Computational Linguistics",
  year = 2020,
  month = "December"
}  
```
