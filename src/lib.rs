#![feature(test)]
#![allow(unused_variables, dead_code, unused_imports)]
use ndarray::parallel::prelude::*;
use ndarray::ArrayD;
use ndarray::{array, s, Array1, Array2, Array3, ArrayView2, ArrayView3, Axis};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::f32::consts::E;
use std::f32::{EPSILON, NEG_INFINITY};

extern crate test;

#[derive(Debug)]
struct Beam {
    prefix: Vec<u32>,
    pr_blank: f32,
    pr_nblank: f32,
}

impl Beam {
    fn new() -> Beam {
        Beam {
            prefix: Vec::new(),
            pr_blank: 1.,
            pr_nblank: 0.,
        }
        // Beam {
        //     prefix: Vec::new(),
        //     pr_blank: 0.,
        //     pr_nblank: NEG_INFINITY,
        // }
    }

    fn pr_total(&self) -> f32 {
        self.pr_blank + self.pr_nblank
    }

    fn score(&self) -> f32 {
        self.pr_total() * (self.prefix.len() as u32 + 1).pow(5) as f32
    }

    fn logsumexp(&self) -> f32 {
        log_sum_exp(array![self.pr_blank, self.pr_nblank])
    }
}

impl Default for Beam {
    fn default() -> Self {
        Beam::new()
    }
}

#[derive(Debug)]
struct Pred {
    p_b: f32,
    p_nb: f32,
}

impl Pred {
    fn new() -> Pred {
        Pred {
            p_b: 0f32,
            p_nb: 0f32,
        }
        // Pred {
        //     p_b: NEG_INFINITY,
        //     p_nb: NEG_INFINITY,
        // }
    }

    fn create(p_b: f32, p_nb: f32) -> Pred {
        Pred {
            p_b: p_b,
            p_nb: p_nb,
        }
    }
}
fn log_sum_exp(arr: Array1<f32>) -> f32 {
    // if arr.iter().all(|x| x.clone() == NEG_INFINITY) {
    //     return NEG_INFINITY;
    // }
    arr.map(|x| x.exp()).sum().log(E)
}

pub fn prefix_beam_search(
    preds: ArrayView2<f32>,
    vocab: &Vec<char>,
    beam_size: u8,
) -> Result<Vec<String>, String> {
    let blank = 0;
    let k = beam_size as usize;
    // let log_preds: Array2<f32> = preds.iter().map(|x| x.log(E)).collect();
    // let log_preds: Array2<f32> = preds.map(|x| x.log(E));
    // println!("{:?}", log_preds);

    let (T, S) = preds.dim();

    let mut beams = vec![Beam::new()];
    for t in 0..T {
        let mut next_beam: HashMap<Vec<u32>, Pred> = HashMap::new();

        // for s in (0..S).filter(|x| preds[[t, x.clone()]] > 0.0000001) {
        for s in 0..S {
            // let p = log_preds[[t, s]];
            let p = preds[[t, s]];

            for beam in &beams {
                let Beam {
                    prefix,
                    pr_blank: p_b,
                    pr_nblank: p_nb,
                } = &beam;
                // println!("{:?}", prefix);
                if s == blank {
                    let curr_prefix_pred = next_beam.entry(prefix.clone()).or_insert(Pred::new());
                    // curr_prefix_pred.p_b =
                    //     log_sum_exp(array![curr_prefix_pred.p_b, p_b + p, p_nb + p]);
                    curr_prefix_pred.p_b += beam.pr_total() * p;
                // println!("{:?}", prefix);
                // continue;
                } else {
                    let mut new_prefix = prefix.clone();
                    new_prefix.push(s as u32);

                    let is_same_as_prev = match prefix.last() {
                        Some(last) => last.clone() == s as u32,
                        None => false,
                    };

                    if new_prefix.len() > 0 && is_same_as_prev {
                        let new_prefix_pred =
                            next_beam.entry(new_prefix.clone()).or_insert(Pred::new());
                        // new_prefix_pred.p_nb =
                        //     log_sum_exp(array![new_prefix_pred.p_nb, p_b + p]);//, p_nb + p]);
                        new_prefix_pred.p_nb += p_b * p;

                        let curr_prefix_pred =
                            next_beam.entry(prefix.clone()).or_insert(Pred::new());
                        // curr_prefix_pred.p_nb =
                        //     log_sum_exp(array![curr_prefix_pred.p_nb, p_nb + p]);
                        curr_prefix_pred.p_nb += p_nb * p;
                    } else {
                        let new_prefix_pred =
                            next_beam.entry(new_prefix.clone()).or_insert(Pred::new());
                        // new_prefix_pred.p_nb =
                        //     log_sum_exp(array![new_prefix_pred.p_nb, p_b + p, p_nb + p]);
                        new_prefix_pred.p_nb += beam.pr_total() * p;
                    }
                }
            }
        }
        // println!("{:?}", next_beam);
        beams = {
            let mut topk_beams = vec![];

            for (prefix, Pred { p_b, p_nb }) in next_beam.iter() {
                let beam = Beam {
                    prefix: prefix.clone(),
                    pr_blank: p_b.clone(),
                    pr_nblank: p_nb.clone(),
                };
                // let score = beam.pr_total();
                // if let Some(Ordering::Greater) = score.partial_cmp(&1.) {
                //     println!("Prefix: {} -> {}",
                //      beam.prefix.iter().map(|x| vocab[x.clone() as usize]).collect::<String>(),
                //     beam.pr_total());
                // }
                topk_beams.push(beam);
            }

            // topk_beams.sort_by_key(|k| k.pr_total());
            topk_beams.sort_by(|a, b| {
                a.pr_total()
                    .partial_cmp(&b.pr_total())
                    .unwrap_or(Ordering::Equal)
            });
            // topk_beams.sort_by(|a, b| a.score().partial_cmp(&b.score()).unwrap());
            // topk_beams.sort_by(|a, b| a.logsumexp().partial_cmp(&b.logsumexp()).unwrap());
            topk_beams.reverse();
            topk_beams.truncate(k);

            // for beam in &topk_beams {
            //     println!("{:?} : {}", beam, beam.logsumexp());
            // }
            topk_beams
        };
        // println!("{:?}", beams);
    }

    // for beam in beams.iter().take(20) {
    //     println!(
    //         "Result: {} -> {}",
    //         beam.prefix
    //             .iter()
    //             .map(|x| vocab[x.clone() as usize])
    //             .collect::<String>(),
    //         beam.pr_total()
    //     );
    // }
    let mut results = vec![];

    for beam in beams.iter().take(20) {
        results.push(
            beam.prefix
                .iter()
                .map(|x| vocab[x.clone() as usize])
                .collect::<String>(),
        );
    }

    // println!("{:?}", beams);
    Ok(results)
}

pub fn prefix_beam_search_batch(
    preds: ArrayView3<f32>,
    vocab: &Vec<char>,
    beam_size: u8,
) -> Result<Vec<Vec<String>>, String> {
    let mut results = Vec::new();

    preds
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| prefix_beam_search(row, vocab, beam_size).unwrap())
        .collect_into_vec(&mut results);

    Ok(results)
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use test::Bencher;
//     use ndarray_npy::read_npy;

//     #[bench]
//     fn bench_add_two(b: &mut Bencher) {

//     let test_2_results = ["10876599",
//         "10876593", "10876594", "10876599n", "10876597", "10876593n", "10876594n", "10876597n", "10876599nn", "10876593nn"];
//     let vocab = ['%', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't', ' '];

//     let maharashtra = "tests/files/MAHARASHTRA.npy";
//     let test2 = "tests/files/test_2.npy";
//     let prestige = "tests/files/PRESTIGE_TOWER.npy";
//     let abcd = "tests/files/ABCD123456.npy";
//     let pred_123456 = "tests/files/123456789012.npy";
//     // let arr : Array2<f32> = read_npy("tests/files/test_2.npy").unwrap();
//     let arr : Array2<f32> = read_npy(prestige).unwrap();

//     b.iter(|| beam_search(&arr, &vocab.to_vec(),  30));
//     }
// }

use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayDyn};
use pyo3::prelude::*;

#[pymodule]
fn beam_search(py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "beam_search")]
    fn beam_search_py(
        _py: Python,
        preds: &PyArray2<f32>,
        vocab: Vec<String>,
        beam_size: u8,
    ) -> PyResult<Vec<String>> {
        let preds = preds.as_array();
        let vocab = vocab
            .iter()
            .map(|x| x.chars().next().unwrap())
            .collect::<Vec<char>>();
        Ok(prefix_beam_search(preds, &vocab, beam_size).unwrap())
    }

    #[pyfn(m, "beam_search_batch")]
    fn beam_search_batch_py(
        _py: Python,
        preds: &PyArray3<f32>,
        vocab: Vec<String>,
        beam_size: u8,
    ) -> PyResult<Vec<Vec<String>>> {
        let preds = preds.as_array();
        let vocab = vocab
            .iter()
            .map(|x| x.chars().next().unwrap())
            .collect::<Vec<char>>();

        Ok(_py.allow_threads(move || prefix_beam_search_batch(preds, &vocab, beam_size).unwrap()))
    }

    Ok(())
}
