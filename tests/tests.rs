use beam_search::*;
use ndarray::{s, Array2, Array3, ArrayD};
use ndarray_npy::read_npy;

#[test]
fn test_beam_search() {
    let test_2_results = [
        "10876599",
        "10876593",
        "10876594",
        "10876599n",
        "10876597",
        "10876593n",
        "10876594n",
        "10876597n",
        "10876599nn",
        "10876593nn",
    ];
    let vocab = [
        '%', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
        'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't', ' ',
    ];

    let maharashtra = "tests/files/MAHARASHTRA.npy";
    let test2 = "tests/files/test_2.npy";
    let prestige = "tests/files/PRESTIGE_TOWER.npy";
    let abcd = "tests/files/ABCD123456.npy";
    let pred_123456 = "tests/files/123456789012.npy";
    // let arr : Array2<f32> = read_npy("tests/files/test_2.npy").unwrap();
    let arr: Array2<f32> = read_npy(maharashtra).unwrap();
    let results = prefix_beam_search(arr.view(), &vocab.to_vec(), 3);

    println!("{:?}", results);

    // println!("{:?}", arr.shape());
    // println!("{:?}", arr.slice(s![1..5, 3..10]));
}

#[test]
fn test_beam_search_batch() {
    let test_2_results = [
        "10876599",
        "10876593",
        "10876594",
        "10876599n",
        "10876597",
        "10876593n",
        "10876594n",
        "10876597n",
        "10876599nn",
        "10876593nn",
    ];
    let vocab = [
        '%', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
        'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't', ' ',
    ];

    let batch1 = "tests/files/Batch_90830534.npy";
    let batch_combined = "tests/files/batch_combined.npy";
    let arr: Array3<f32> = read_npy(batch_combined).unwrap();
    let results = prefix_beam_search_batch(arr.view(), &vocab.to_vec(), 3);

    println!("{:?}", results);

    // println!("{:?}", arr.shape());
    // println!("{:?}", arr.slice(s![1..5, 3..10]));
}
