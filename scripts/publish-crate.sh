#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

readonly expected_package='ps-hayro-jpeg2000'
readonly expected_registry='crates-io'

[ "${GITHUB_REF:-}" = 'refs/heads/master' ] || fail 'publishing is allowed only from master'
[ "${LOCAL_CHECKS_ATTESTED:-}" = 'yes' ] || fail 'local checks were not attested'
[ -n "${CARGO_REGISTRY_TOKEN:-}" ] || fail 'CARGO_REGISTRY_TOKEN is required'

expected_commit=${EXPECTED_COMMIT:-}
case "$expected_commit" in
  ''|*[!0-9a-f]*) fail 'EXPECTED_COMMIT must be a full lowercase hexadecimal SHA' ;;
esac
[ "${#expected_commit}" -eq 40 ] || fail 'EXPECTED_COMMIT must be a full 40-character commit SHA'
actual_commit=$(git rev-parse HEAD)
[ "$actual_commit" = "$expected_commit" ] || fail 'checked out commit does not match EXPECTED_COMMIT'
if ! worktree_status=$(git status --porcelain --untracked-files=all); then
  fail 'unable to inspect working tree status'
fi
[ -z "$worktree_status" ] || fail 'working tree must be clean'

# Read only simple package fields from the root manifest. This avoids an extra
# JSON/TOML parsing dependency in the publishing workflow.
manifest_value() {
  local field=$1
  awk -F= -v field="$field" '
    /^\[package\][[:space:]]*$/ { in_package = 1; next }
    /^\[/ { if (in_package) exit; next }
    in_package {
      key = $1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
      if (key == field) {
        value = substr($0, index($0, "=") + 1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        print value
        exit
      }
    }
  ' Cargo.toml
}

package_name=$(manifest_value name | sed -E 's/^"([^"]*)"$/\1/')
version=$(manifest_value version | sed -E 's/^"([^"]*)"$/\1/')
publish_allowlist=$(manifest_value publish | tr -d '[:space:]')
[ "$package_name" = "$expected_package" ] || fail "Cargo.toml package.name must be $expected_package"
[ -n "$version" ] || fail 'Cargo.toml package.version is required'
case "$version" in
  *[!0-9A-Za-z.+-]*) fail 'Cargo.toml package.version contains unsafe URL characters' ;;
esac
[ "$publish_allowlist" = '["crates-io"]' ] || fail 'Cargo.toml package.publish must allow crates-io only'

version_id=${version//+/%2B}
status=$(curl --disable --silent --show-error \
  --user-agent 'manifest-cargo-publisher (https://github.com/pathscale/ps-hayro-jpeg2000)' \
  --connect-timeout 10 --max-time 30 --retry 0 \
  --output /dev/null --write-out '%{http_code}' \
  "https://crates.io/api/v1/crates/$package_name/$version_id") \
  || fail 'crates.io registry request failed'

case "$status" in
  200)
    printf '%s %s is already published on crates.io; nothing to do\n' "$package_name" "$version"
    exit 0
    ;;
  404)
    printf '%s %s is absent from crates.io; publishing\n' "$package_name" "$version"
    ;;
  *)
    fail "crates.io returned HTTP $status; refusing to treat it as an absent version"
    ;;
esac

if ! worktree_status=$(git status --porcelain --untracked-files=all); then
  fail 'unable to inspect working tree status'
fi
[ -z "$worktree_status" ] || fail 'working tree must be clean'
cargo publish --registry "$expected_registry" --package "$package_name"
