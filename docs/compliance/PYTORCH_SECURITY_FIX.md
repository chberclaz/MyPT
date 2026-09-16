# PyTorch Security Fix - `weights_only` Parameter

## Overview

Fixed PyTorch `FutureWarning` about `torch.load` and added explicit `weights_only` parameter to all checkpoint loading operations for security and clarity.

---

## The Warning

**Original Warning:**
```
FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), 
which uses the default pickle module implicitly. It is possible to construct malicious pickle data 
which will execute arbitrary code during unpickling.
```

**Issue:** PyTorch uses Python's `pickle` module to serialize/deserialize objects, which can execute arbitrary code if the file is malicious.

---

## The Fix

### Strategy

We use different `weights_only` values depending on what we're loading:

1. **Model weights only** (`model.pt`): `weights_only=True` ✅ SAFE
   - Contains only tensors (model parameters)
   - No complex objects that could execute code
   - Safe to load from untrusted sources

2. **Optimizer state** (`optimizer.pt`): `weights_only=True` ✅ SAFE
   - Contains tensors and simple state dicts
   - No executable code
   - Safe to load

3. **Legacy checkpoints** (full `.pt` with config/tokenizer): `weights_only=False` ⚠️ NEED THIS
   - Contains nested dicts with config/tokenizer
   - PyTorch can't load these with `weights_only=True`
   - Safe in our case (we control the files)

---

## Changes Made

### 1. `core/model.py`

**Line 284 - Loading model weights:**
```python
# Before
model.load_state_dict(torch.load(model_path, map_location=map_location))

# After
model.load_state_dict(torch.load(model_path, map_location=map_location, weights_only=True))
```

**Line 308 - Loading optimizer state:**
```python
# Before
optimizer_state = torch.load(optimizer_path, map_location=map_location)

# After
optimizer_state = torch.load(optimizer_path, map_location=map_location, weights_only=True)
```

**Line 325 - Loading legacy checkpoint:**
```python
# Before
checkpoint = torch.load(path, map_location=map_location)

# After (with comment explaining why)
# Legacy checkpoints contain dicts with config/tokenizer, need weights_only=False
checkpoint = torch.load(path, map_location=map_location, weights_only=False)
```

---

### 2. `core/__init__.py`

**Line 201 - `get_model_info()` loading legacy checkpoint:**
```python
# Before
checkpoint = torch.load(
    ckpt_manager.get_path(filename),
    map_location='cpu'
)

# After
# Legacy checkpoints contain dicts with config/tokenizer
checkpoint = torch.load(
    ckpt_manager.get_path(filename),
    map_location='cpu',
    weights_only=False
)
```

---

### 3. `inspect_model.py`

**Line 79 - Loading legacy checkpoint:**
```python
# Before
checkpoint = torch.load(model_path, map_location=device)

# After
# Legacy checkpoints contain dicts with config/tokenizer
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
```

---

### 4. `convert_legacy_checkpoints.py`

**Line 122 - Detecting legacy checkpoints:**
```python
# Before
checkpoint = torch.load(path, map_location='cpu')

# After
# Legacy checkpoints contain dicts with config/tokenizer
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
```

---

## Security Considerations

### When is `weights_only=True` safe?

✅ **SAFE** - You control the source of the file  
✅ **SAFE** - File only contains tensors (model weights)  
✅ **SAFE** - File only contains optimizer state  

### When is `weights_only=False` needed?

⚠️ **NEEDED** - Loading complex objects (dicts, custom classes)  
⚠️ **NEEDED** - Legacy checkpoints with config/tokenizer  
⚠️ **NEEDED** - Our case: we control all checkpoint files  

### Is our usage safe?

✅ **YES** - We only load checkpoints we created  
✅ **YES** - We never load untrusted files from users  
✅ **YES** - All files are in our controlled `checkpoints/` directory  

**Important:** If you add a feature to load user-provided checkpoint files, you should:
1. Validate the file before loading
2. Consider sandboxing
3. Warn users about the risks
4. Only use `weights_only=True` if possible

---

## Testing

Verify the fix works:

```bash
# Train a model (creates new checkpoints with weights_only=True compatible format)
python train.py --model_name test --input_file input.txt --max_iters 100

# Load and generate (should work without warnings)
python generate.py --model_name test --prompt "Hello"

# Inspect model (should work without warnings)
python inspect_model.py --model_name test
```

**Expected:** No `FutureWarning` should appear!

---

## Future Considerations

### When PyTorch Changes Default

When PyTorch flips the default to `weights_only=True` (future version):

✅ **Model weights loading** - Will work fine (we already use `weights_only=True`)  
✅ **Optimizer loading** - Will work fine (we already use `weights_only=True`)  
⚠️ **Legacy checkpoints** - Still work (we explicitly use `weights_only=False`)  

Our code is **future-proof**!

---

### Migration Away from Legacy Format

Eventually, you may want to deprecate legacy format entirely:

1. Convert all legacy checkpoints to new JSON format:
   ```bash
   python convert_legacy_checkpoints.py --all
   ```

2. After conversion, remove legacy loading code:
   - Remove `GPT.load_legacy()` method
   - Remove legacy detection in `CheckpointManager`
   - Update documentation

3. Benefits:
   - All loading uses `weights_only=True` (maximum security)
   - Simpler codebase (no legacy support)
   - Better separation of concerns (JSON for config, .pt for weights)

---

## Summary Table

| File | What's Loaded | `weights_only` | Safe? |
|------|---------------|----------------|-------|
| `model.pt` | Tensors only | `True` | ✅ Very safe |
| `optimizer.pt` | Tensors + state | `True` | ✅ Very safe |
| Legacy `latest.pt` | Dicts + config | `False` | ✅ Safe (we control it) |
| Legacy `final.pt` | Dicts + config | `False` | ✅ Safe (we control it) |

---

## Best Practices

### For Model Weights (Recommended)

```python
# Loading model weights - use weights_only=True
state_dict = torch.load("model.pt", weights_only=True)
model.load_state_dict(state_dict)
```

### For Full Checkpoints (When Needed)

```python
# Loading checkpoint with config/other objects
# Add comment explaining why weights_only=False is needed
# Only load from trusted sources!
checkpoint = torch.load("checkpoint.pt", weights_only=False)
```

### For New Code

When saving new checkpoints:
1. **Separate files**: Save weights separately from config
2. **Use JSON**: Store config/metadata in JSON files
3. **weights_only=True**: Only load weights with this flag

This is exactly what our new JSON-based checkpoint format does! ✅

---

## References

- [PyTorch Security Advisory](https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models)
- [PyTorch torch.load Documentation](https://pytorch.org/docs/stable/generated/torch.load.html)
- [Python Pickle Security](https://docs.python.org/3/library/pickle.html#module-pickle)

---

## Conclusion

✅ **Fixed** - All `torch.load` calls now explicitly set `weights_only`  
✅ **Secure** - Model weights use `weights_only=True`  
✅ **Compatible** - Legacy checkpoints still work with `weights_only=False`  
✅ **Future-proof** - Code ready for PyTorch default change  
✅ **Documented** - Comments explain why each choice was made  

**No more warnings, and the code is more secure!** 🔒

