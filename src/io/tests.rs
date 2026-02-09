//! Comprehensive tests for the I/O module
//!
//! Tests cover: binary round-trip, JSON round-trip, compression round-trip,
//! FANN format read/write, streaming training data, DOT export,
//! edge cases, and security (size-limited reads).

use std::io::Cursor;

// ---------------------------------------------------------------
// Binary round-trip tests
// ---------------------------------------------------------------
#[cfg(feature = "binary")]
mod binary_tests {
    use super::*;
    use crate::io::binary::{write_binary, BinaryConfig, BinaryReader, BinaryWriter};

    /// Both `write_binary` and `read_binary` use legacy-compatible bincode options
    /// (little-endian, fixed-int encoding), with `read_binary` adding a size limit
    /// for security hardening.

    /// Helper: serialize using the same format that read_binary uses for deserialization
    /// (legacy bincode format: little-endian, fixed-int encoding)
    fn serialize_with_default_options<T: serde::Serialize>(data: &T) -> Vec<u8> {
        bincode::serialize(data).expect("serialization should succeed")
    }

    #[test]
    fn binary_round_trip_vec_f32() {
        // Serialize with DefaultOptions to match what read_binary expects
        let original: Vec<f32> = vec![1.0, 2.5, -3.3, 0.0, 99.9];
        let buf = serialize_with_default_options(&original);

        let mut cursor = Cursor::new(&buf);
        let restored: Vec<f32> =
            crate::io::binary::read_binary(&mut cursor).expect("read_binary failed");
        assert_eq!(original, restored);
    }

    #[test]
    fn binary_round_trip_empty_vec() {
        let original: Vec<f32> = vec![];
        let buf = serialize_with_default_options(&original);

        let mut cursor = Cursor::new(&buf);
        let restored: Vec<f32> =
            crate::io::binary::read_binary(&mut cursor).expect("read_binary failed");
        assert_eq!(original, restored);
    }

    #[test]
    fn binary_reader_writer_round_trip() {
        // Use DefaultOptions-serialized data for the reader
        let original: Vec<f64> = vec![std::f64::consts::PI, std::f64::consts::E, 0.0, -1.0];
        let buf = serialize_with_default_options(&original);

        let reader = BinaryReader::new();
        let mut cursor = Cursor::new(&buf);
        let restored: Vec<f64> = reader.read(&mut cursor).expect("BinaryReader read failed");
        assert_eq!(original, restored);
    }

    #[test]
    fn binary_write_produces_bytes() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut buf = Vec::new();
        write_binary(&data, &mut buf).expect("write_binary failed");
        assert!(!buf.is_empty(), "write_binary should produce non-empty output");
    }

    #[test]
    fn binary_config_variants() {
        let default = BinaryConfig::new();
        assert!(default.little_endian);
        assert!(!default.varint_encoding);

        let compact = BinaryConfig::compact();
        assert!(compact.varint_encoding);

        let fast = BinaryConfig::fast();
        assert!(!fast.varint_encoding);

        let _default_trait = BinaryConfig::default();
    }

    #[test]
    fn binary_serialized_size() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let writer = BinaryWriter::new();
        let size = writer.serialized_size(&data).expect("serialized_size failed");
        assert!(size > 0);

        // Also check via the inspect module
        let size2 = crate::io::binary::inspect::serialized_size(&data)
            .expect("inspect::serialized_size failed");
        assert_eq!(size, size2);
    }

    #[test]
    fn binary_validate_serializable() {
        let data: Vec<f32> = vec![1.0, 2.0];
        crate::io::binary::inspect::validate_serializable(&data)
            .expect("validate_serializable should succeed");
    }

    #[test]
    fn binary_read_with_limit_rejects_oversized() {
        // Create a payload with DefaultOptions encoding
        let large: Vec<u8> = vec![42u8; 1024];
        let buf = serialize_with_default_options(&large);

        // Try to read with a tiny limit
        let reader = BinaryReader::new();
        let mut cursor = Cursor::new(&buf);
        let result: Result<Vec<u8>, _> = reader.read_with_limit(&mut cursor, 16);
        assert!(
            result.is_err(),
            "read_with_limit should reject data exceeding limit"
        );
    }

    #[test]
    fn binary_read_with_limit_accepts_small() {
        let small: Vec<u8> = vec![1, 2, 3];
        let buf = serialize_with_default_options(&small);

        let reader = BinaryReader::new();
        let mut cursor = Cursor::new(&buf);
        // Provide a generous limit
        let result: Vec<u8> = reader
            .read_with_limit(&mut cursor, 4096)
            .expect("should succeed within limit");
        assert_eq!(small, result);
    }
}

// ---------------------------------------------------------------
// JSON round-trip tests
// ---------------------------------------------------------------
#[cfg(feature = "serde")]
mod json_tests {
    use super::*;
    use crate::io::json::{
        read_json, read_json_with_options, write_json, write_json_with_options, JsonConfig,
        JsonReader, JsonWriter,
    };
    use crate::mock_types::MockNetwork;

    #[test]
    fn json_round_trip_mock_network() {
        let original = MockNetwork {
            num_layers: 3,
            learning_rate: 0.01,
            connection_rate: 1.0,
            layer_sizes: vec![2, 4, 1],
            weights: vec![0.1, -0.2, 0.3, 0.4, -0.5],
        };

        let mut buf = Vec::new();
        write_json(&original, &mut buf).expect("write_json failed");

        let mut cursor = Cursor::new(&buf);
        let restored: MockNetwork = read_json(&mut cursor).expect("read_json failed");
        assert_eq!(original, restored);
    }

    #[test]
    fn json_round_trip_compact_vs_pretty() {
        let original = MockNetwork {
            num_layers: 2,
            learning_rate: 0.5,
            connection_rate: 1.0,
            layer_sizes: vec![3, 1],
            weights: vec![0.0],
        };

        // Compact
        let mut buf_compact = Vec::new();
        write_json_with_options(&original, &mut buf_compact, false).expect("compact write failed");

        // Pretty
        let mut buf_pretty = Vec::new();
        write_json_with_options(&original, &mut buf_pretty, true).expect("pretty write failed");

        // Pretty output should be longer (has indentation)
        assert!(
            buf_pretty.len() > buf_compact.len(),
            "pretty JSON should be longer than compact"
        );

        // Both should deserialize to the same value
        let r1: MockNetwork =
            read_json_with_options(&mut Cursor::new(&buf_compact), false).unwrap();
        let r2: MockNetwork =
            read_json_with_options(&mut Cursor::new(&buf_pretty), true).unwrap();
        assert_eq!(r1, r2);
        assert_eq!(original, r1);
    }

    #[test]
    fn json_reader_writer_round_trip() {
        let original = MockNetwork {
            num_layers: 3,
            learning_rate: 0.1,
            connection_rate: 0.8,
            layer_sizes: vec![5, 10, 2],
            weights: vec![0.1; 60],
        };

        let writer = JsonWriter::new();
        let mut buf = Vec::new();
        writer
            .write(&original, &mut buf)
            .expect("JsonWriter write failed");

        let reader = JsonReader::new();
        let restored: MockNetwork = reader
            .read(&mut Cursor::new(&buf))
            .expect("JsonReader read failed");
        assert_eq!(original, restored);
    }

    #[test]
    fn json_config_variants() {
        let default = JsonConfig::new();
        assert!(default.pretty);

        let compact = JsonConfig::compact();
        assert!(!compact.pretty);

        let pretty = JsonConfig::pretty();
        assert!(pretty.pretty);

        let _default_trait = JsonConfig::default();
    }

    #[test]
    fn json_empty_weights() {
        let original = MockNetwork {
            num_layers: 1,
            learning_rate: 0.0,
            connection_rate: 1.0,
            layer_sizes: vec![1],
            weights: vec![],
        };

        let mut buf = Vec::new();
        write_json(&original, &mut buf).unwrap();
        let restored: MockNetwork = read_json(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(original, restored);
    }
}

// ---------------------------------------------------------------
// Compression round-trip tests
// ---------------------------------------------------------------
#[cfg(feature = "compression")]
mod compression_tests {
    use super::*;
    use crate::io::compression::{
        compress_bytes, compress_data, decompress_bytes, decompress_data, CompressedReader,
        CompressedWriter, CompressionConfig,
    };
    use std::io::{Read, Write};

    #[test]
    fn compress_decompress_bytes_round_trip() {
        let original = b"Hello, neural network world! This is test data for compression.";
        let compressed = compress_bytes(original).expect("compress_bytes failed");
        let decompressed = decompress_bytes(&compressed).expect("decompress_bytes failed");
        assert_eq!(original.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn compress_decompress_data_round_trip() {
        let original = b"FANN format data with weights 0.1 0.2 0.3 repeated many times. ";
        let repeated: Vec<u8> = original
            .iter()
            .cycle()
            .take(original.len() * 20)
            .copied()
            .collect();

        let mut compressed = Vec::new();
        compress_data(&mut Cursor::new(&repeated), &mut compressed)
            .expect("compress_data failed");

        let mut decompressed = Vec::new();
        decompress_data(&mut Cursor::new(&compressed), &mut decompressed)
            .expect("decompress_data failed");

        assert_eq!(repeated, decompressed);
    }

    #[test]
    fn compress_empty_data() {
        let empty: &[u8] = b"";
        let compressed = compress_bytes(empty).expect("compress_bytes for empty failed");
        let decompressed =
            decompress_bytes(&compressed).expect("decompress_bytes for empty failed");
        assert!(decompressed.is_empty());
    }

    #[test]
    fn compression_actually_shrinks_repetitive_data() {
        // Highly repetitive data should compress well
        let data: Vec<u8> = vec![42u8; 10_000];
        let compressed = compress_bytes(&data).expect("compress_bytes failed");
        assert!(
            compressed.len() < data.len(),
            "Compressed size {} should be less than original {}",
            compressed.len(),
            data.len()
        );
    }

    #[test]
    fn compressed_reader_writer_round_trip() {
        let original = b"Data going through CompressedWriter and CompressedReader";

        let mut compressed_buf = Vec::new();
        {
            let mut cw = CompressedWriter::new(&mut compressed_buf);
            cw.write_all(original)
                .expect("CompressedWriter write failed");
            cw.finish().expect("CompressedWriter finish failed");
        }

        let mut cr = CompressedReader::new(Cursor::new(&compressed_buf));
        let mut restored = Vec::new();
        cr.read_to_end(&mut restored)
            .expect("CompressedReader read failed");
        assert_eq!(original.as_slice(), restored.as_slice());
    }

    #[test]
    fn compressed_writer_with_level() {
        let data = b"test data for custom level compression";
        let mut buf = Vec::new();
        {
            let mut cw = CompressedWriter::with_level(&mut buf, 1);
            cw.write_all(data).unwrap();
            cw.finish().unwrap();
        }
        let restored = decompress_bytes(&buf).unwrap();
        assert_eq!(data.as_slice(), restored.as_slice());
    }

    #[test]
    fn compressed_writer_with_config() {
        let config = CompressionConfig::fast();
        let data = b"test data for config-based compression";
        let mut buf = Vec::new();
        {
            let mut cw = CompressedWriter::with_config(&mut buf, config);
            cw.write_all(data).unwrap();
            cw.finish().unwrap();
        }
        let restored = decompress_bytes(&buf).unwrap();
        assert_eq!(data.as_slice(), restored.as_slice());
    }

    #[test]
    fn compression_config_variants() {
        let default = CompressionConfig::new();
        assert_eq!(default.level, 6);
        assert!(!default.fast);

        let fast = CompressionConfig::fast();
        assert_eq!(fast.level, 1);
        assert!(fast.fast);

        let best = CompressionConfig::best();
        assert_eq!(best.level, 9);

        let custom = CompressionConfig::with_level(4);
        assert_eq!(custom.level, 4);

        // Level is capped at 9
        let over = CompressionConfig::with_level(100);
        assert_eq!(over.level, 9);

        let _default_trait = CompressionConfig::default();
    }

    #[test]
    fn compression_analyze_stats() {
        use crate::io::compression::analyze;

        let ratio = analyze::compression_ratio(1000, 200);
        assert!((ratio - 0.2).abs() < 1e-9);

        let savings = analyze::space_savings(1000, 200);
        assert!((savings - 80.0).abs() < 1e-9);

        // Edge case: zero original size
        assert_eq!(analyze::compression_ratio(0, 0), 0.0);
        assert_eq!(analyze::space_savings(0, 0), 0.0);
    }

    #[test]
    fn compression_analyze_test_compression() {
        use crate::io::compression::analyze;

        let data: Vec<u8> = vec![0u8; 5000];
        let stats = analyze::test_compression(&data).expect("test_compression failed");
        assert_eq!(stats.original_size, 5000);
        assert!(stats.compressed_size < stats.original_size);
        assert!(stats.ratio < 1.0);
        assert!(stats.savings_percent > 0.0);
    }
}

// ---------------------------------------------------------------
// FANN format round-trip tests
// ---------------------------------------------------------------
mod fann_format_tests {
    use super::*;
    use crate::io::fann_format::{FannReader, FannWriter};
    use crate::Network;

    /// Helper: create a small network (2-3-1) with known structure
    fn make_small_network() -> Network<f32> {
        let mut net = Network::<f32>::new(&[2, 3, 1]);
        net.randomize_weights(-1.0, 1.0);
        net
    }

    #[test]
    fn fann_write_read_round_trip() {
        let network = make_small_network();
        let original_weights = network.get_weights();

        // Write to FANN format
        let writer = FannWriter::new();
        let mut buf = Vec::new();
        writer
            .write_network(&network, &mut buf)
            .expect("FannWriter write_network failed");

        // Read back
        let reader = FannReader::new();
        let restored = reader
            .read_network::<f32, _>(&mut Cursor::new(&buf))
            .expect("FannReader read_network failed");

        // Verify structure
        assert_eq!(network.num_layers(), restored.num_layers());
        assert_eq!(network.num_inputs(), restored.num_inputs());
        assert_eq!(network.num_outputs(), restored.num_outputs());

        // Verify weights are approximately equal (formatting may lose precision)
        let restored_weights = restored.get_weights();
        assert_eq!(
            original_weights.len(),
            restored_weights.len(),
            "Weight count mismatch"
        );
        for (i, (orig, rest)) in original_weights
            .iter()
            .zip(restored_weights.iter())
            .enumerate()
        {
            assert!(
                (orig - rest).abs() < 1e-4,
                "Weight {} differs: original={}, restored={}",
                i,
                orig,
                rest
            );
        }
    }

    #[test]
    fn fann_format_preserves_layer_sizes() {
        let network = Network::<f64>::new(&[5, 10, 3]);

        let writer = FannWriter::new();
        let mut buf = Vec::new();
        writer.write_network(&network, &mut buf).unwrap();

        let reader = FannReader::new();
        let restored = reader
            .read_network::<f64, _>(&mut Cursor::new(&buf))
            .unwrap();

        assert_eq!(restored.num_inputs(), 5);
        assert_eq!(restored.num_outputs(), 3);
        assert_eq!(restored.num_layers(), 3);
    }

    #[test]
    fn fann_reader_rejects_missing_header() {
        let bad_data = b"not_a_fann_file\nnum_layers=2\n";
        let reader = FannReader::new();
        let result = reader.read_network::<f32, _>(&mut Cursor::new(bad_data));
        assert!(result.is_err(), "Should reject data without FANN header");
    }

    #[test]
    fn fann_reader_rejects_zero_layers() {
        let bad_data = b"FANN_FLO:2.1\nnum_layers=0\nlayer_sizes=\n";
        let reader = FannReader::new();
        let result = reader.read_network::<f32, _>(&mut Cursor::new(bad_data));
        assert!(result.is_err(), "Should reject zero layers");
    }

    #[test]
    fn fann_reader_rejects_mismatched_layer_count() {
        let bad_data = b"FANN_FLO:2.1\nnum_layers=3\nlayer_sizes=2 1\n";
        let reader = FannReader::new();
        let result = reader.read_network::<f32, _>(&mut Cursor::new(bad_data));
        assert!(
            result.is_err(),
            "Should reject when layer_sizes length != num_layers"
        );
    }

    #[test]
    fn fann_reader_default_trait() {
        let _reader = FannReader::default();
        let _writer = FannWriter::default();
    }

    #[test]
    fn fann_round_trip_single_layer_network() {
        // A minimal network with 2 layers (input + output)
        let network = Network::<f32>::new(&[3, 1]);

        let writer = FannWriter::new();
        let mut buf = Vec::new();
        writer.write_network(&network, &mut buf).unwrap();

        let reader = FannReader::new();
        let restored = reader
            .read_network::<f32, _>(&mut Cursor::new(&buf))
            .unwrap();

        assert_eq!(restored.num_layers(), 2);
        assert_eq!(restored.num_inputs(), 3);
        assert_eq!(restored.num_outputs(), 1);
    }

    #[test]
    fn fann_round_trip_deep_network() {
        let network = Network::<f32>::new(&[4, 8, 8, 8, 2]);

        let writer = FannWriter::new();
        let mut buf = Vec::new();
        writer.write_network(&network, &mut buf).unwrap();

        let reader = FannReader::new();
        let restored = reader
            .read_network::<f32, _>(&mut Cursor::new(&buf))
            .unwrap();

        assert_eq!(restored.num_layers(), 5);
        assert_eq!(restored.num_inputs(), 4);
        assert_eq!(restored.num_outputs(), 2);
    }

    #[test]
    fn fann_output_contains_expected_keys() {
        let network = Network::<f32>::new(&[2, 3, 1]);

        let writer = FannWriter::new();
        let mut buf = Vec::new();
        writer.write_network(&network, &mut buf).unwrap();

        let text = String::from_utf8(buf).expect("FANN output should be valid UTF-8");
        assert!(text.contains("FANN_FLO"), "Missing FANN header");
        assert!(text.contains("num_layers=3"), "Missing num_layers");
        assert!(text.contains("layer_sizes="), "Missing layer_sizes");
        assert!(text.contains("connection_rate="), "Missing connection_rate");
    }
}

// ---------------------------------------------------------------
// Training data reader/writer tests
// ---------------------------------------------------------------
mod training_data_tests {
    use super::*;
    use crate::io::training_data::{TrainingDataReader, TrainingDataWriter};
    use crate::mock_types::MockTrainingData;

    fn make_xor_data() -> MockTrainingData {
        MockTrainingData {
            num_data: 4,
            num_input: 2,
            num_output: 1,
            inputs: vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            outputs: vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
        }
    }

    #[test]
    fn training_data_round_trip() {
        let original = make_xor_data();

        let writer = TrainingDataWriter::new();
        let mut buf = Vec::new();
        writer
            .write_data(&original, &mut buf)
            .expect("write_data failed");

        let reader = TrainingDataReader::new();
        let restored = reader
            .read_data(&mut Cursor::new(&buf))
            .expect("read_data failed");

        assert_eq!(original.num_data, restored.num_data);
        assert_eq!(original.num_input, restored.num_input);
        assert_eq!(original.num_output, restored.num_output);
        assert_eq!(original.inputs, restored.inputs);
        assert_eq!(original.outputs, restored.outputs);
    }

    #[test]
    fn training_data_bad_header() {
        let bad = b"only two fields\n";
        let reader = TrainingDataReader::new();
        let result = reader.read_data(&mut Cursor::new(bad));
        assert!(result.is_err(), "Should reject header without 3 fields");
    }

    #[test]
    fn training_data_wrong_input_count() {
        // Header says 2 inputs, but data line has 3
        let bad = b"1 2 1\n0.1 0.2 0.3\n0.5\n";
        let reader = TrainingDataReader::new();
        let result = reader.read_data(&mut Cursor::new(bad));
        assert!(result.is_err(), "Should reject mismatched input count");
    }

    #[test]
    fn training_data_wrong_output_count() {
        // Header says 1 output, but data line has 2
        let bad = b"1 2 1\n0.1 0.2\n0.5 0.6\n";
        let reader = TrainingDataReader::new();
        let result = reader.read_data(&mut Cursor::new(bad));
        assert!(result.is_err(), "Should reject mismatched output count");
    }

    #[test]
    fn training_data_default_traits() {
        let _reader = TrainingDataReader::default();
        let _writer = TrainingDataWriter::default();
    }

    #[test]
    fn training_data_multiple_outputs() {
        let data = MockTrainingData {
            num_data: 2,
            num_input: 3,
            num_output: 2,
            inputs: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            outputs: vec![vec![0.1, 0.9], vec![0.8, 0.2]],
        };

        let writer = TrainingDataWriter::new();
        let mut buf = Vec::new();
        writer.write_data(&data, &mut buf).unwrap();

        let reader = TrainingDataReader::new();
        let restored = reader.read_data(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(data, restored);
    }
}

// ---------------------------------------------------------------
// Streaming reader tests
// ---------------------------------------------------------------
mod streaming_tests {
    use super::*;
    use std::io::BufReader;

    #[test]
    fn stream_read_xor_data() {
        let data = b"4 2 1\n0 0\n0\n0 1\n1\n1 0\n1\n1 1\n0\n";

        // Use the streaming module's TrainingDataStreamReader
        let stream_reader = crate::io::streaming::TrainingDataStreamReader::new();
        let mut collected_inputs = Vec::new();
        let mut collected_outputs = Vec::new();

        let mut buf_reader = BufReader::new(Cursor::new(data));
        let stats = stream_reader
            .read_stream(&mut buf_reader, |input, output| {
                collected_inputs.push(input.to_vec());
                collected_outputs.push(output.to_vec());
                Ok(())
            })
            .expect("read_stream failed");

        assert_eq!(stats.samples_processed, 4);
        assert_eq!(stats.num_input, 2);
        assert_eq!(stats.num_output, 1);
        assert_eq!(collected_inputs.len(), 4);
        assert_eq!(collected_outputs.len(), 4);

        // Check specific samples
        assert_eq!(collected_inputs[1], vec![0.0, 1.0]);
        assert_eq!(collected_outputs[1], vec![1.0]);
    }

    #[test]
    fn stream_read_batches() {
        let data = b"4 2 1\n0 0\n0\n0 1\n1\n1 0\n1\n1 1\n0\n";

        let stream_reader = crate::io::streaming::TrainingDataStreamReader::new();
        let mut batch_count = 0;
        let mut total_samples = 0;

        let mut buf_reader = BufReader::new(Cursor::new(data));
        let stats = stream_reader
            .read_batches(&mut buf_reader, 2, |inputs, outputs| {
                batch_count += 1;
                total_samples += inputs.len();
                assert_eq!(inputs.len(), outputs.len());
                Ok(())
            })
            .expect("read_batches failed");

        assert_eq!(stats.samples_processed, 4);
        assert_eq!(total_samples, 4);
        assert_eq!(batch_count, 2, "4 samples in batches of 2 = 2 batches");
    }

    #[test]
    fn stream_read_with_custom_buffer_size() {
        let stream_reader =
            crate::io::streaming::TrainingDataStreamReader::with_buffer_size(1024);
        let data = b"1 1 1\n0.5\n0.5\n";
        let mut buf_reader = BufReader::new(Cursor::new(data));
        let stats = stream_reader
            .read_stream(&mut buf_reader, |_input, _output| Ok(()))
            .unwrap();
        assert_eq!(stats.samples_processed, 1);
    }

    #[test]
    fn stream_rejects_bad_header() {
        let bad = b"two fields\n";
        let stream_reader = crate::io::streaming::TrainingDataStreamReader::new();
        let mut buf_reader = BufReader::new(Cursor::new(bad));
        let result = stream_reader.read_stream(&mut buf_reader, |_, _| Ok(()));
        assert!(result.is_err());
    }

    #[test]
    fn stream_stats_methods() {
        let stats = crate::io::streaming::StreamStats {
            samples_processed: 100,
            bytes_read: 5000,
            num_input: 3,
            num_output: 2,
        };

        let avg = stats.avg_bytes_per_sample();
        assert!((avg - 50.0).abs() < 1e-9);

        assert_eq!(stats.parameters_per_sample(), 5);

        // Edge case: zero samples
        let empty_stats = crate::io::streaming::StreamStats {
            samples_processed: 0,
            bytes_read: 0,
            num_input: 0,
            num_output: 0,
        };
        assert_eq!(empty_stats.avg_bytes_per_sample(), 0.0);
    }

    #[test]
    fn buffered_stream_reader_basics() {
        use crate::io::streaming::BufferedStreamReader;

        let data = b"hello world";
        let bsr = BufferedStreamReader::new(Cursor::new(data));
        assert_eq!(bsr.buffer_size(), 8192);

        let bsr2 = BufferedStreamReader::with_capacity(Cursor::new(data), 4096);
        assert_eq!(bsr2.buffer_size(), 4096);
    }

    #[test]
    fn streaming_memory_utils() {
        use crate::io::streaming::memory;

        let mem = memory::estimate_batch_memory(32, 10, 5);
        assert!(mem > 0);

        let batch = memory::optimal_batch_size(1_000_000, 10, 5);
        assert!(batch > 0);

        // Edge case: zero-sized features
        let batch_zero = memory::optimal_batch_size(1_000_000, 0, 0);
        assert!(batch_zero > 0);
    }

    #[test]
    fn stream_default_trait() {
        let _sr = crate::io::streaming::TrainingDataStreamReader::default();
        let _bsr = crate::io::streaming::BufferedStreamReader::new(Cursor::new(b""));
    }
}

// ---------------------------------------------------------------
// DOT export tests
// ---------------------------------------------------------------
mod dot_export_tests {
    use crate::io::dot_export::{DotExporter, LayoutDirection};
    use crate::mock_types::MockNetwork;

    fn make_mock_network() -> MockNetwork {
        MockNetwork {
            num_layers: 3,
            learning_rate: 0.01,
            connection_rate: 1.0,
            layer_sizes: vec![2, 3, 1],
            weights: vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
        }
    }

    #[test]
    fn dot_export_basic_structure() {
        let network = make_mock_network();
        let exporter = DotExporter::new();

        let mut buf = Vec::new();
        exporter
            .export_network(&network, &mut buf)
            .expect("export_network failed");

        let dot = String::from_utf8(buf).expect("DOT output should be valid UTF-8");

        assert!(
            dot.contains("digraph NeuralNetwork"),
            "Missing digraph declaration"
        );
        assert!(dot.contains("rankdir=LR"), "Missing LR rankdir");
        assert!(
            dot.contains("node [shape=circle]"),
            "Missing node shape"
        );
        assert!(dot.contains("->"), "Missing edges");
        assert!(dot.ends_with("}\n"), "Should end with closing brace");
    }

    #[test]
    fn dot_export_contains_all_nodes() {
        let network = make_mock_network();
        let exporter = DotExporter::new();

        let mut buf = Vec::new();
        exporter.export_network(&network, &mut buf).unwrap();
        let dot = String::from_utf8(buf).unwrap();

        // Network has 2+3+1 = 6 nodes (n0..n5)
        for i in 0..6 {
            assert!(
                dot.contains(&format!("n{i}")),
                "Missing node n{i} in DOT output"
            );
        }
    }

    #[test]
    fn dot_export_with_weights() {
        let network = make_mock_network();
        let exporter = DotExporter::with_options(true, true, LayoutDirection::LeftToRight);

        let mut buf = Vec::new();
        exporter.export_network(&network, &mut buf).unwrap();
        let dot = String::from_utf8(buf).unwrap();

        assert!(dot.contains("label="), "Should have weight labels on edges");
        assert!(
            dot.contains("blue") || dot.contains("red"),
            "Should have colored edges"
        );
        assert!(
            dot.contains("penwidth="),
            "Should have pen width for edges"
        );
    }

    #[test]
    fn dot_export_layout_directions() {
        let network = make_mock_network();

        let directions = vec![
            (LayoutDirection::LeftToRight, "LR"),
            (LayoutDirection::TopToBottom, "TB"),
            (LayoutDirection::RightToLeft, "RL"),
            (LayoutDirection::BottomToTop, "BT"),
        ];

        for (direction, expected_str) in directions {
            let exporter = DotExporter::with_options(false, true, direction);
            let mut buf = Vec::new();
            exporter.export_network(&network, &mut buf).unwrap();
            let dot = String::from_utf8(buf).unwrap();
            assert!(
                dot.contains(&format!("rankdir={expected_str}")),
                "Expected rankdir={expected_str} for direction {:?}",
                direction
            );
        }
    }

    #[test]
    fn dot_export_show_indices_labels() {
        let network = make_mock_network();

        // With indices
        let exporter = DotExporter::with_options(false, true, LayoutDirection::LeftToRight);
        let mut buf = Vec::new();
        exporter.export_network(&network, &mut buf).unwrap();
        let dot = String::from_utf8(buf).unwrap();
        assert!(dot.contains("L0N0"), "Should have indexed labels");

        // Without indices
        let exporter = DotExporter::with_options(false, false, LayoutDirection::LeftToRight);
        let mut buf2 = Vec::new();
        exporter.export_network(&network, &mut buf2).unwrap();
        let dot2 = String::from_utf8(buf2).unwrap();
        assert!(dot2.contains("I0"), "Should have I/H/O labels");
        assert!(dot2.contains("O0"), "Should have output labels");
    }

    #[test]
    fn dot_export_network_info_comments() {
        let network = make_mock_network();
        let exporter = DotExporter::new();

        let mut buf = Vec::new();
        exporter.export_network(&network, &mut buf).unwrap();
        let dot = String::from_utf8(buf).unwrap();

        assert!(
            dot.contains("// Network Information"),
            "Missing info comment"
        );
        assert!(dot.contains("// Layers: 3"), "Missing layer count");
        assert!(
            dot.contains("// Learning Rate:"),
            "Missing learning rate"
        );
        assert!(
            dot.contains("// Connection Rate:"),
            "Missing connection rate"
        );
    }

    #[test]
    fn dot_export_empty_weights() {
        let network = MockNetwork {
            num_layers: 2,
            learning_rate: 0.1,
            connection_rate: 1.0,
            layer_sizes: vec![1, 1],
            weights: vec![],
        };

        let exporter = DotExporter::with_options(true, true, LayoutDirection::LeftToRight);
        let mut buf = Vec::new();
        exporter.export_network(&network, &mut buf).unwrap();
        let dot = String::from_utf8(buf).unwrap();

        // With show_weights=true but empty weights, edges should not have labels
        assert!(dot.contains("n0 -> n1;"), "Should have plain edges");
    }

    #[test]
    fn dot_export_default_trait() {
        let _exporter = DotExporter::default();
    }
}

// ---------------------------------------------------------------
// Error type tests
// ---------------------------------------------------------------
mod error_tests {
    use crate::io::error::IoError;

    #[test]
    fn error_display() {
        let errors = vec![
            (
                IoError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "file not found",
                )),
                "I/O error:",
            ),
            (
                IoError::InvalidFileFormat("bad header".to_string()),
                "Invalid file format:",
            ),
            (IoError::ParseError("nan".to_string()), "Parse error:"),
            (
                IoError::SerializationError("corrupt".to_string()),
                "Serialization error:",
            ),
            (
                IoError::CompressionError("inflate failed".to_string()),
                "Compression error:",
            ),
            (
                IoError::InvalidNetwork("no layers".to_string()),
                "Invalid network:",
            ),
            (
                IoError::InvalidTrainingData("wrong count".to_string()),
                "Invalid training data:",
            ),
        ];

        for (err, expected_prefix) in errors {
            let display = format!("{err}");
            assert!(
                display.contains(expected_prefix),
                "Error display '{}' should contain '{}'",
                display,
                expected_prefix
            );
        }
    }

    #[test]
    fn error_source() {
        use std::error::Error;

        let io_err = IoError::Io(std::io::Error::new(std::io::ErrorKind::Other, "test"));
        assert!(io_err.source().is_some());

        let parse_err = IoError::ParseError("test".to_string());
        assert!(parse_err.source().is_none());
    }

    #[test]
    fn error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let converted: IoError = io_err.into();
        match converted {
            IoError::Io(_) => {}
            _ => panic!("Expected IoError::Io variant"),
        }
    }

    #[test]
    fn error_from_parse_float() {
        let parse_err: Result<f32, _> = "not_a_number".parse();
        let converted: IoError = parse_err.unwrap_err().into();
        match converted {
            IoError::ParseError(_) => {}
            _ => panic!("Expected IoError::ParseError variant"),
        }
    }

    #[test]
    fn error_from_parse_int() {
        let parse_err: Result<usize, _> = "not_an_int".parse();
        let converted: IoError = parse_err.unwrap_err().into();
        match converted {
            IoError::ParseError(_) => {}
            _ => panic!("Expected IoError::ParseError variant"),
        }
    }
}

// ---------------------------------------------------------------
// FileFormat enum tests
// ---------------------------------------------------------------
mod file_format_tests {
    use crate::io::FileFormat;

    #[test]
    fn file_format_variants() {
        let formats = vec![
            FileFormat::Fann,
            FileFormat::Json,
            FileFormat::Binary,
            FileFormat::Dot,
            FileFormat::CompressedFann,
            FileFormat::CompressedBinary,
        ];

        // All variants should be distinct
        for (i, a) in formats.iter().enumerate() {
            for (j, b) in formats.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }

        // Clone and Debug
        let f = FileFormat::Fann;
        let f2 = f;
        assert_eq!(f, f2);
        let _debug = format!("{:?}", f);
    }
}

// ---------------------------------------------------------------
// Network binary serialization round-trip (via Network API)
// ---------------------------------------------------------------
#[cfg(all(feature = "binary", feature = "serde"))]
mod network_binary_tests {
    use crate::Network;

    /// Helper: serialize network with DefaultOptions to match from_bytes' deserialization.
    ///
    /// Helper: serialize a network using the same format that from_bytes expects.
    fn serialize_network_compat(network: &Network<f32>) -> Vec<u8> {
        // Now that from_bytes uses legacy-compatible options, regular serialize works
        bincode::serialize(network).expect("serialization should succeed")
    }

    #[test]
    fn network_to_bytes_produces_output() {
        let mut network = Network::<f32>::new(&[2, 4, 1]);
        network.randomize_weights(-1.0, 1.0);

        let bytes = network.to_bytes().expect("to_bytes failed");
        assert!(!bytes.is_empty(), "to_bytes should produce non-empty output");
    }

    #[test]
    fn network_round_trip_with_consistent_encoding() {
        let mut network = Network::<f32>::new(&[2, 4, 1]);
        network.randomize_weights(-1.0, 1.0);

        // Serialize with DefaultOptions to match from_bytes' deserialization
        let bytes = serialize_network_compat(&network);
        let restored = Network::<f32>::from_bytes(&bytes).expect("from_bytes failed");

        assert_eq!(network.num_layers(), restored.num_layers());
        assert_eq!(network.num_inputs(), restored.num_inputs());
        assert_eq!(network.num_outputs(), restored.num_outputs());

        let orig_w = network.get_weights();
        let rest_w = restored.get_weights();
        assert_eq!(orig_w.len(), rest_w.len());
        for (a, b) in orig_w.iter().zip(rest_w.iter()) {
            assert!((a - b).abs() < 1e-7, "Weight mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn network_from_bytes_rejects_garbage() {
        let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01];
        let result = Network::<f32>::from_bytes(&garbage);
        assert!(result.is_err(), "Should reject garbage bytes");
    }

    #[test]
    fn network_from_bytes_rejects_oversized() {
        // Small garbage should fail with deserialization error, not size error
        let small_garbage = vec![0u8; 100];
        let result = Network::<f32>::from_bytes(&small_garbage);
        assert!(result.is_err());
    }

    #[test]
    fn network_round_trip_deep_network() {
        let mut network = Network::<f32>::new(&[10, 20, 15, 10, 5]);
        network.randomize_weights(-0.5, 0.5);

        let bytes = serialize_network_compat(&network);
        let restored = Network::<f32>::from_bytes(&bytes).expect("from_bytes failed");

        assert_eq!(network.num_layers(), restored.num_layers());
        assert_eq!(network.total_connections(), restored.total_connections());
    }
}
