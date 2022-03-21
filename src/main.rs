use std::{mem::size_of_val, time::Instant};

use zksnark::groth16::{
    circuit::{ASTParser, TryParse},
    coefficient_poly::CoefficientPoly,
    fr::FrLocal,
    QAP,
};

use bellman::{
    gadgets::{
        boolean::{AllocatedBit, Boolean},
        multipack,
        sha256::sha256,
    },
    Circuit, ConstraintSystem, SynthesisError,
};
use bls12_381::Bls12;
use ff::PrimeField;
use jemalloc_ctl::{epoch, stats};
use rand::thread_rng;
use sha2::{Digest, Sha256};

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn sha256_le<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
    mut cs: CS,
    data: &[Boolean],
) -> Result<Vec<Boolean>, SynthesisError> {
    // Flip endianness of each input byte
    let input: Vec<_> = data
        .chunks(8)
        .map(|c| c.iter().rev())
        .flatten()
        .cloned()
        .collect();

    let res = sha256(cs.namespace(|| "SHA-256(input)"), &input)?;

    // Flip endianness of each output byte
    Ok(res
        .chunks(8)
        .map(|c| c.iter().rev())
        .flatten()
        .cloned()
        .collect())
}

struct MyCircuit {
    /// The input to SHA-256d we are proving that we know. Set to `None` when we
    /// are verifying a proof (and do not have the witness data).
    preimage: Option<[u8; 80]>,
}

impl<Scalar: PrimeField> Circuit<Scalar> for MyCircuit {
    fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        // Compute the values for the bits of the preimage. If we are verifying a proof,
        // we still need to create the same constraints, so we return an equivalent-size
        // Vec of None (indicating that the value of each bit is unknown).
        let bit_values = if let Some(preimage) = self.preimage {
            preimage
                .into_iter()
                .map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
                .flatten()
                .map(|b| Some(b))
                .collect()
        } else {
            vec![None; 80 * 8]
        };
        assert_eq!(bit_values.len(), 80 * 8);

        // Witness the bits of the preimage.
        let preimage_bits = bit_values
            .into_iter()
            .enumerate()
            // Allocate each bit.
            .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {}", i)), b))
            // Convert the AllocatedBits into Booleans (required for the sha256 gadget).
            .map(|b| b.map(Boolean::from))
            .collect::<Result<Vec<_>, _>>()?;

        // Compute hash = SHA-256d(preimage).
        let hash = sha256_le(cs.namespace(|| "SHA-256(preimage)"), &preimage_bits)?;

        // Expose the vector of 32 boolean variables as compact public inputs.
        multipack::pack_into_inputs(cs.namespace(|| "pack hash"), &hash)
    }
}

fn zksnark_example() {
    epoch::advance().unwrap();

    println!("-- zksnark simple quadratic example");
    let quadratic_zk_code = r#"
        (in x a b c)
        (out y)
        (verify x y)

        (program
            (= xsqur
                (* x x))
            (= axsqur
                (* a xsqur))
            (= bx
                (* b x))
            (= y
                (+ axsqur bx c)))
    "#;

    let qap: QAP<CoefficientPoly<FrLocal>> =
        ASTParser::try_parse(quadratic_zk_code).unwrap().into();

    let assignments = &[
        0.into(), // x
        1.into(), // a
        1.into(), // b
        0.into(), // c
    ];

    let weights = zksnark::groth16::weights(quadratic_zk_code, assignments).unwrap();
    let (sigmag1, sigmag2) = zksnark::groth16::setup(&qap);

    let allocated_pre_proof_gen = stats::allocated::read().unwrap();

    let mut begin = Instant::now();
    let proof = zksnark::groth16::prove(&qap, (&sigmag1, &sigmag2), &weights);
    println!(
        "time (proof generation): {}",
        humantime::format_duration(begin.elapsed()).to_string()
    );
    println!("size (proof): {} bytes", size_of_val(&proof));

    epoch::advance().unwrap();
    let allocated = stats::allocated::read().unwrap();
    println!(
        "memory (proof generation): {} bytes",
        allocated - allocated_pre_proof_gen
    );

    begin = Instant::now();

    // Verify the statement "I know x and y such that x^2 + x == y"
    let verification_result = zksnark::groth16::verify(
        &qap,
        (sigmag1, sigmag2),
        &vec![FrLocal::from(0), FrLocal::from(0)],
        proof,
    );
    println!(
        "time (proof verification): {}",
        humantime::format_duration(begin.elapsed()).to_string()
    );
    println!("verification result: {}", verification_result);
    assert!(verification_result);
}

fn bellman_example() {
    epoch::advance().unwrap();

    println!("\n-- bellman SHA256 example");

    let mut begin = Instant::now();

    // Create parameters for our circuit. In a production deployment these would
    // be generated securely using a multiparty computation.
    let params = {
        let c = MyCircuit { preimage: None };
        bellman::groth16::generate_random_parameters::<Bls12, _, _>(c, &mut thread_rng()).unwrap()
    };
    println!(
        "time (parameter generation): {}",
        humantime::format_duration(begin.elapsed()).to_string()
    );
    begin = Instant::now();

    // Prepare the verification key (for proof verification).
    let pvk = bellman::groth16::prepare_verifying_key(&params.vk);
    println!(
        "time (verification key generation): {}",
        humantime::format_duration(begin.elapsed()).to_string()
    );
    begin = Instant::now();

    // Pick a preimage and compute its hash.
    let preimage = [42; 80];
    let hash = Sha256::digest(&preimage);

    // Create an instance of our circuit (with the preimage as a witness).
    let c = MyCircuit {
        preimage: Some(preimage),
    };

    let allocated_pre_proof_gen = stats::allocated::read().unwrap();

    // Create a Groth16 proof with our parameters.
    let proof = bellman::groth16::create_random_proof(c, &params, &mut thread_rng()).unwrap();
    println!(
        "time (proof generation): {}",
        humantime::format_duration(begin.elapsed()).to_string()
    );
    println!("size (proof): {} bytes", size_of_val(&proof));

    epoch::advance().unwrap();
    let allocated = stats::allocated::read().unwrap();
    println!(
        "memory (proof generation): {} bytes",
        allocated - allocated_pre_proof_gen
    );
    begin = Instant::now();

    // Pack the hash as inputs for proof verification.
    let hash_bits = multipack::bytes_to_bits_le(&hash);
    let inputs = multipack::compute_multipacking(&hash_bits);

    let verification_result = bellman::groth16::verify_proof(&pvk, &proof, &inputs);
    println!(
        "time (proof verification): {}",
        humantime::format_duration(begin.elapsed()).to_string()
    );
    println!("verification result: {}", verification_result.is_ok());
    assert!(verification_result.is_ok());
}

#[test]
fn zksnark_example_test() {
    zksnark_example();
}

#[test]
fn bellman_example_test() {
    bellman_example();
}

fn main() {
    zksnark_example();
    bellman_example();
}
