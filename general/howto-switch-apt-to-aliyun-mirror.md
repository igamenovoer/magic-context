# How to point apt to the Aliyun mirror (Ubuntu version agnostic)

Use this when you need faster package downloads inside mainland China or on Alibaba Cloud. The approach updates every apt source file so it works with both the traditional `/etc/apt/sources.list` layout (Ubuntu ≤ 23.10) and the newer DEB822 format introduced in Ubuntu 24.04. The same pattern can also switch to the Tsinghua TUNA mirror.

## Steps

1. **Back up existing lists** – keep a rollback copy before editing.
   ```bash
   sudo cp -a /etc/apt/sources.list /etc/apt/sources.list.bak
   sudo cp -a /etc/apt/sources.list.d /etc/apt/sources.list.d.bak 2>/dev/null || true
   ```

2. **Replace mirror hosts everywhere** – swap only the host portion so the distribution codename (`$(lsb_release -cs)`) stays intact.
   ```bash
   MIRROR=https://mirrors.aliyun.com   # for Tsinghua TUNA use https://mirrors.tuna.tsinghua.edu.cn
   TARGETS="/etc/apt/sources.list"
   if [ -d /etc/apt/sources.list.d ]; then
     TARGETS="${TARGETS} $(find /etc/apt/sources.list.d -type f \( -name '*.list' -o -name '*.sources' \))"
   fi
   for file in ${TARGETS}; do
     [ -f "$file" ] || continue
     sudo sed -i "s@https\?://[^[:space:]\"']*archive\.ubuntu\.com@${MIRROR}@g" "$file"
     sudo sed -i "s@https\?://[^[:space:]\"']*security\.ubuntu\.com@${MIRROR}@g" "$file"
   done
   ```
   - For Alibaba Cloud ECS instances, change `https://mirrors.aliyun.com` to `http://mirrors.cloud.aliyuncs.com` as required by the platform.

3. **Refresh package metadata** – confirm apt connects to the new mirror.
   ```bash
   sudo apt clean
   sudo apt update
   ```

4. **Optional: verify** – reinstall a small package and ensure the download URLs point at the mirror host (e.g., `mirrors.aliyun.com` or `mirrors.tuna.tsinghua.edu.cn`).

## References

- Aliyun mirror overview and configuration guidance: <https://developer.aliyun.com/mirror/ubuntu>
- Aliyun instructions for replacing archive and security mirrors via `sed`: <https://developer.aliyun.com/mirror/ubuntu-releases/>
- Tsinghua University (TUNA) mirror usage guide with sed replacement examples: <https://www.iotcolon.com/?p=555>
