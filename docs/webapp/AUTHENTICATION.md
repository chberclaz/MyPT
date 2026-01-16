# MyPT Authentication System

Comprehensive documentation for the MyPT web application authentication system.

## Overview

MyPT implements a minimal but secure authentication system designed for offline, air-gapped environments. The system uses:

- **JWT (JSON Web Tokens)** for stateless session management
- **bcrypt** for secure password hashing
- **Role-based access control (RBAC)** with two user levels
- **File-based user storage** for simplicity and offline operation

---

## User Roles & Privileges

### Role: `user`

Standard user with access to inference and chat features.

| Resource                            | Access                         |
| ----------------------------------- | ------------------------------ |
| **Pages**                           |                                |
| `/login`                            | ✅ Allowed                     |
| `/chat`                             | ✅ Allowed                     |
| `/training`                         | ❌ Denied (redirects to login) |
| **Chat API**                        |                                |
| `GET /api/chat/models`              | ✅ Allowed                     |
| `GET /api/chat/history`             | ✅ Allowed                     |
| `POST /api/chat/send`               | ✅ Allowed                     |
| `POST /api/chat/clear`              | ✅ Allowed                     |
| **Workspace API**                   |                                |
| `GET /api/workspace/info`           | ✅ Allowed                     |
| `GET /api/workspace/documents`      | ✅ Allowed                     |
| `POST /api/workspace/rebuild-index` | ❌ Denied                      |
| **Training API**                    |                                |
| `GET /api/training/status`          | ❌ Denied                      |
| `POST /api/training/start`          | ❌ Denied                      |
| `POST /api/training/stop`           | ❌ Denied                      |
| `WS /api/training/ws`               | ❌ Denied                      |
| **Debug API**                       |                                |
| `GET /api/debug/status`             | ❌ Denied                      |
| `POST /api/debug/toggle`            | ❌ Denied                      |

### Role: `admin`

Administrator with full access to all features including training.

| Resource                            | Access     |
| ----------------------------------- | ---------- |
| **Pages**                           |            |
| `/login`                            | ✅ Allowed |
| `/chat`                             | ✅ Allowed |
| `/training`                         | ✅ Allowed |
| **Chat API**                        |            |
| `GET /api/chat/models`              | ✅ Allowed |
| `GET /api/chat/history`             | ✅ Allowed |
| `POST /api/chat/send`               | ✅ Allowed |
| `POST /api/chat/clear`              | ✅ Allowed |
| **Workspace API**                   |            |
| `GET /api/workspace/info`           | ✅ Allowed |
| `GET /api/workspace/documents`      | ✅ Allowed |
| `POST /api/workspace/rebuild-index` | ✅ Allowed |
| **Training API**                    |            |
| `GET /api/training/status`          | ✅ Allowed |
| `POST /api/training/start`          | ✅ Allowed |
| `POST /api/training/stop`           | ✅ Allowed |
| `WS /api/training/ws`               | ✅ Allowed |
| **Debug API**                       |            |
| `GET /api/debug/status`             | ✅ Allowed |
| `POST /api/debug/toggle`            | ✅ Allowed |

### Public Endpoints (No Authentication Required)

| Endpoint      | Description                 |
| ------------- | --------------------------- |
| `GET /login`  | Login page                  |
| `POST /login` | Login form submission       |
| `GET /logout` | Logout and clear session    |
| `GET /health` | Health check for monitoring |

---

## Security Features

### Password Hashing

Passwords are hashed using **bcrypt** via the `passlib` library:

- Adaptive cost factor (automatically adjusts to hardware)
- Built-in salt generation (unique per password)
- Timing-attack resistant comparison
- Industry-standard algorithm recommended by OWASP

If `passlib` is not available, the system falls back to PBKDF2-HMAC-SHA256 with:

- 100,000 iterations
- Random 32-character salt
- Constant-time comparison

### JWT Tokens

Sessions are managed using JWT tokens stored in HTTP-only cookies:

| Property       | Value            |
| -------------- | ---------------- |
| Algorithm      | HS256            |
| Token Lifetime | 24 hours         |
| Storage        | HTTP-only cookie |
| Cookie Name    | `mypt_session`   |
| SameSite       | Lax              |

**Security Properties:**

- **HTTP-only**: Token cannot be accessed by JavaScript (XSS protection)
- **SameSite=Lax**: Prevents CSRF attacks on state-changing requests
- **Signed**: Tokens are cryptographically signed to prevent tampering

### Secret Key

The JWT signing key is:

1. Read from environment variable `MYPT_SECRET_KEY` if set
2. Auto-generated as a random 256-bit hex string if not set

**For production deployments**, always set a persistent secret key:

```bash
# Generate a secure key
python -c "import secrets; print(secrets.token_hex(32))"

# Set in environment
export MYPT_SECRET_KEY="your-generated-key-here"
```

### Fallback Mode

If `python-jose` is not installed, the system uses a simple signed token format:

- Base64-encoded JSON payload
- HMAC-SHA256 signature
- Same security properties for the use case

---

## User Management

### User Storage

Users are stored in `webapp/users.json`:

```json
{
  "admin": {
    "hashed_password": "$2b$12$...",
    "role": "admin"
  },
  "user": {
    "hashed_password": "$2b$12$...",
    "role": "user"
  }
}
```

### Default Users

On first run, the system automatically creates default users:

| Username | Password | Role  |
| -------- | -------- | ----- |
| `admin`  | `admin`  | admin |
| `user`   | `user`   | user  |

⚠️ **WARNING**: Change these credentials immediately in production!

### Creating Users

#### Method 1: Python Script

```python
from webapp.auth import create_user

# Create a new user
create_user("alice", "secure_password_123", role="user")

# Create a new admin
create_user("bob", "admin_password_456", role="admin")
```

#### Method 2: Direct JSON Editing

1. Generate a password hash:

```python
from webapp.auth import pwd_context
hashed = pwd_context.hash("my_secure_password")
print(hashed)
```

2. Add to `webapp/users.json`:

```json
{
  "existing_users": "...",
  "new_user": {
    "hashed_password": "$2b$12$paste_hash_here",
    "role": "user"
  }
}
```

#### Method 3: CLI Helper (Create if needed)

```python
# save as scripts/manage_users.py
import sys
sys.path.insert(0, '.')
from webapp.auth import create_user, change_password, load_users, save_users

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manage MyPT users")
    parser.add_argument("action", choices=["add", "delete", "list", "passwd"])
    parser.add_argument("--username", "-u")
    parser.add_argument("--password", "-p")
    parser.add_argument("--role", "-r", default="user", choices=["user", "admin"])

    args = parser.parse_args()

    if args.action == "add":
        if create_user(args.username, args.password, args.role):
            print(f"Created user: {args.username} (role: {args.role})")
        else:
            print(f"User already exists: {args.username}")

    elif args.action == "delete":
        users = load_users()
        if args.username in users:
            del users[args.username]
            save_users(users)
            print(f"Deleted user: {args.username}")
        else:
            print(f"User not found: {args.username}")

    elif args.action == "list":
        users = load_users()
        print("Users:")
        for name, data in users.items():
            print(f"  - {name} (role: {data['role']})")

    elif args.action == "passwd":
        if change_password(args.username, args.password):
            print(f"Password changed for: {args.username}")
        else:
            print(f"User not found: {args.username}")

if __name__ == "__main__":
    main()
```

Usage:

```bash
# List users
python scripts/manage_users.py list

# Add user
python scripts/manage_users.py add -u alice -p secret123 -r user

# Add admin
python scripts/manage_users.py add -u bob -p admin456 -r admin

# Change password
python scripts/manage_users.py passwd -u alice -p newpassword

# Delete user
python scripts/manage_users.py delete -u alice
```

### Deleting Users

#### Method 1: Python

```python
from webapp.auth import load_users, save_users

users = load_users()
if "username_to_delete" in users:
    del users["username_to_delete"]
    save_users(users)
```

#### Method 2: Direct JSON Editing

1. Open `webapp/users.json`
2. Remove the user entry
3. Save the file

### Changing Passwords

```python
from webapp.auth import change_password

change_password("username", "new_secure_password")
```

---

## Configuration

### Environment Variables

| Variable          | Description       | Default                         |
| ----------------- | ----------------- | ------------------------------- |
| `MYPT_SECRET_KEY` | JWT signing key   | Random (regenerated on restart) |
| `MYPT_DEBUG`      | Enable debug mode | `false`                         |

### Code Constants (in `webapp/auth.py`)

| Constant             | Description         | Default        |
| -------------------- | ------------------- | -------------- |
| `TOKEN_EXPIRE_HOURS` | Session duration    | 24 hours       |
| `COOKIE_NAME`        | Session cookie name | `mypt_session` |
| `ALGORITHM`          | JWT algorithm       | `HS256`        |

### Customization

To customize token expiration:

```python
# In webapp/auth.py
TOKEN_EXPIRE_HOURS = 8  # 8-hour sessions
```

To enable secure cookies (HTTPS only):

```python
# In webapp/auth.py, in create_login_response()
response.set_cookie(
    ...
    secure=True,  # Only send over HTTPS
    ...
)
```

---

## API Reference

### Authentication Endpoints

#### POST /login

Authenticate and create session.

**Request (Form Data):**

```
username=admin&password=admin&next=/chat
```

**Success Response:**

- Status: 302 Redirect to `next` URL
- Cookie: `mypt_session` set with JWT token

**Error Response:**

- Re-renders login page with error message

#### GET /logout

End session and clear cookie.

**Response:**

- Status: 302 Redirect to `/login`
- Cookie: `mypt_session` deleted

#### GET /api/auth/me

Get current user information.

**Request Headers:**

```
Cookie: mypt_session=<jwt_token>
```

**Response:**

```json
{
  "username": "admin",
  "role": "admin",
  "is_admin": true
}
```

---

## Troubleshooting

### "Not authenticated" Error

**Cause:** Session expired or invalid token.

**Solution:** Log in again at `/login`.

### "Admin access required" Error

**Cause:** User role is `user`, not `admin`.

**Solution:** Log in with an admin account, or upgrade user role in `users.json`.

### Default Users Not Created

**Cause:** `webapp/users.json` already exists but is empty or corrupted.

**Solution:** Delete `webapp/users.json` and restart the application.

### Password Not Working After Restart

**Cause:** `MYPT_SECRET_KEY` changed (tokens signed with old key are invalid).

**Solution:**

1. Set a persistent `MYPT_SECRET_KEY` environment variable
2. Or simply log in again

### bcrypt Not Available

**Warning Message:**

```
Using fallback password hashing (PBKDF2)
```

**Solution:** Install passlib with bcrypt:

```bash
pip install passlib[bcrypt]
```

---

## Security Best Practices

### For Production Deployment

1. **Change default passwords immediately**

   ```python
   from webapp.auth import change_password
   change_password("admin", "very_secure_password_here")
   change_password("user", "another_secure_password")
   ```

2. **Set a persistent secret key**

   ```bash
   export MYPT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
   ```

3. **Use HTTPS** (if deploying on network)

   - Configure a reverse proxy (nginx, caddy) with TLS
   - Set `secure=True` in cookie settings

4. **Limit admin accounts**

   - Only create admin accounts for users who need training access
   - Regular users should have `role: user`

5. **Regular password rotation**

   - Change passwords periodically
   - Remove unused accounts

6. **Audit logging** (future enhancement)
   - Log all authentication events
   - Monitor for failed login attempts

### Security Limitations

This authentication system is designed for **internal, trusted environments**:

- No rate limiting on login attempts
- No account lockout after failed attempts
- No two-factor authentication
- No password complexity requirements
- Session tokens are not revocable (expire naturally)

For high-security deployments, consider:

- Adding rate limiting via reverse proxy
- Implementing 2FA
- Using an external identity provider
- Adding session revocation (requires server-side state)

---

## Dependencies

### Required

- `fastapi` - Web framework
- `passlib[bcrypt]` - Password hashing (recommended)
- `python-jose[cryptography]` - JWT handling (recommended)

### Fallback Support

The system works without optional dependencies:

- Without `passlib`: Uses PBKDF2-HMAC-SHA256
- Without `python-jose`: Uses HMAC-signed base64 tokens

Install all dependencies:

```bash
pip install passlib[bcrypt] python-jose[cryptography]
```

---

## File Structure

```
webapp/
├── auth.py              # Authentication module
├── users.json           # User database (auto-created)
├── main.py              # Routes with auth dependencies
├── routers/
│   ├── chat.py          # Protected with require_user
│   ├── training.py      # Protected with require_admin
│   └── workspace.py     # Mixed (user/admin)
└── templates/
    ├── login.html       # Login page
    └── base.html        # User badge in header
```

---

## Version History

| Version | Date     | Changes                               |
| ------- | -------- | ------------------------------------- |
| 1.0     | Dec 2024 | Initial authentication implementation |

---

## See Also

- [WEBAPP_GUIDE.md](WEBAPP_GUIDE.md) - Full web application documentation
- [README.md](../README.md) - Project overview
