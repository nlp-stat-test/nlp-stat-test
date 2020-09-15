const fileLabelId = "fileLabel"
const MAX_FILE_SIZE_IN_KB = 100
const MAX_NUM_SAMPLES = 10000
// 'invalid' to disallow files over MAX_FILE_SIZE_IN_KB, 'warn' to just warn
const LARGE_FILE_ACTION =  'warn' // or 'invalid'
const FILE_TOO_LARGE_MESSAGE =
    "Error: The selected file must not be larger than 100 kB"
const WARN_MESSAGE =
    '<br> &nbsp;<span style="color:tomato;"><i>Warning: your file is over the size limit. The calculation will sample from a subset of the data.</i></span>'