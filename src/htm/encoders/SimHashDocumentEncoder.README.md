# SimHash Document Encoder Algorithm Details

SimHash is a Locality-Sensitive Hashing (LSH) algorithm from the world of
nearest-neighbor document similarity search. It is used by the GoogleBot Web
Crawler to find near-duplicate web pages.

"Similarity" here refers to bitwise similarity (small hamming distance, high
overlap), not semantic similarity (encodings for "apple" and "computer" will
have no relation here). Internally, hamming distances are never considered or
adjusted -- they're always the result of a kind of dynamic statistical
distribution.


## Code

| What | Where |
| ---- | ----- |
| Source | `./SimHashDocumentEncoder.hpp` |
| Example | `py/htm/examples/encoders/simhash_document_encoder.py` |
| Tests | `bindings/py/tests/encoders/simhash_document_encoder_test.py` |


## Example

> **Note on Hashing:**
Imagine below that we are using a tiny hashing function which outputs a 9-bit
long binary string digest. (In reality, we use the SHA3 hashing function, with
the SHAKE256 XOF to give us a binary string digest output which is the same
length as our output SDR encoding.)

Let's say you have a document: **["hello", "there", "world"]**.

Tokens:

```
hello
there
world
```


### Example when `tokenSimilarity` is OFF:

#### Hash Tokens

```
hello => [ 0 1 0 1 0 1 0 1 1 ]
there => [ 0 0 0 1 0 0 1 0 0 ]
world => [ 1 0 0 1 0 1 0 1 0 ]
```

#### Prepare Token Hashes for Summation

Replace all binary `0` values with signed integer `-1`. Weighting could also
take place during this step if desired.

```
hello => [ -1 +1 -1 +1 -1 +1 -1 +1 +1 ]
there => [ -1 -1 -1 +1 -1 -1 +1 -1 -1 ]
world => [ +1 -1 -1 +1 -1 +1 -1 +1 -1 ]
```

#### Document Summation

```
sums  => [ -1 -1 -3 +3 -3 +1 -1 +1 -1 ]
```

#### Document SimHash

We choose the top N sums for our sparse results. (In a regular non-sparse
SimHash, we would filter on >= 0).

```
sim   => [  0  0  0  1  0  1  0  1  0 ]
```


### Example when `tokenSimilarity` ON:

#### Hash Tokens *AND individual Characters*

```
hello => [ 0 1 0 1 0 1 0 1 1 ]
    h => [ 1 0 1 1 0 1 0 0 1 ]
    e => [ 0 1 0 1 1 0 0 0 1 ]
    l => [ 1 1 0 1 1 0 1 0 0 ]
    l => [ 1 1 0 1 1 0 1 0 0 ]
    o => [ 1 1 1 0 0 0 0 0 0 ]
there => [ 0 0 0 1 0 0 1 0 0 ]
    t => [ 1 1 1 0 1 0 1 0 1 ]
    h => [ 1 0 1 1 0 1 0 0 1 ]
    e => [ 0 1 0 1 1 0 0 0 1 ]
    r => [ 0 0 0 1 1 0 0 0 0 ]
    e => [ 0 1 0 1 1 0 0 0 1 ]
world => [ 1 0 0 1 0 1 0 1 0 ]
    w => [ 0 0 0 0 1 1 1 0 1 ]
    o => [ 1 1 1 0 0 0 0 0 0 ]
    r => [ 0 0 1 1 0 0 0 1 1 ]
    l => [ 1 1 0 1 1 0 1 0 0 ]
    d => [ 0 1 0 0 1 0 0 0 0 ]
```

#### Prepare Token/Char Hashes for Summation

Replace all binary `0` values with signed integer `-1`. Weighting could also
take place during this step if desired.

```
hello => [ -1 +1 -1 +1 -1 +1 -1 +1 +1 ]
    h => [ +1 -1 +1 +1 -1 +1 -1 -1 +1 ]
    e => [ -1 +1 -1 +1 +1 -1 -1 -1 +1 ]
    l => [ +1 +1 -1 +1 +1 -1 +1 -1 -1 ]
    l => [ +1 +1 -1 +1 +1 -1 +1 -1 -1 ]
    o => [ +1 +1 +1 -1 -1 -1 -1 -1 -1 ]
there => [ -1 -1 -1 +1 -1 -1 +1 -1 -1 ]
    t => [ +1 +1 +1 -1 +1 -1 +1 -1 +1 ]
    h => [ +1 -1 +1 +1 -1 +1 -1 -1 +1 ]
    e => [ -1 +1 -1 +1 +1 -1 -1 -1 +1 ]
    r => [ -1 -1 -1 +1 +1 -1 -1 -1 -1 ]
    e => [ -1 +1 -1 +1 +1 -1 -1 -1 +1 ]
world => [ +1 -1 -1 +1 -1 +1 -1 +1 -1 ]
    w => [ -1 -1 -1 -1 +1 +1 +1 -1 +1 ]
    o => [ +1 +1 +1 -1 -1 -1 -1 -1 -1 ]
    r => [ -1 -1 +1 +1 -1 -1 -1 +1 +1 ]
    l => [ +1 +1 -1 +1 +1 -1 +1 -1 -1 ]
    d => [ -1 +1 -1 -1 +1 -1 -1 -1 -1 ]
```

#### Document Summation

```
sums  => [ +0 +2 -6 +8 -2 -8 -6 -12 +0 ]
```

#### Document SimHash

We choose the top N sums for our sparse results. (In a regular non-sparse
SimHash, we would filter on >= 0).

```
sim   => [  1  1  0  1  0  0  0  0  1  ]
```


## Learn More

| What | Where |
| ---- | ----- |
| [Locality-Sensitive Hashing Article](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) | Wikipedia |
| [SimHash Article](https://en.wikipedia.org/wiki/SimHash) | Wikipedia |
| [SimHashing Video](https://www.youtube.com/watch?v=gnraT4N43qo) | YouTube |
