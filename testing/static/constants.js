const fileLabelId = "fileLabel"
const euSizeId = 'eval_unit_size'
const MAX_FILE_SIZE_IN_KB = 10000  // 10000 for 10MB
const MAX_NUM_SAMPLES = 100000
const MAX_FILE_SIZE = '10 MB'
// 'invalid' to disallow files over MAX_FILE_SIZE_IN_KB, 'warn' to just warn
const LARGE_FILE_ACTION =  'invalid' // or 'warn'
const FILE_TOO_LARGE_MESSAGE =
    "Error: The selected file must not be larger than " + MAX_FILE_SIZE
const WARN_MESSAGE =
    '<br> &nbsp;<span style="color:tomato;"><i>Warning: your file is over the '+ MAX_FILE_SIZE
    ' size limit. ' +
    '<br> See the manual for instructions on running NLPStats locally.</i></span>'