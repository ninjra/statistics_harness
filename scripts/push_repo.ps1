# Push current branch to origin using WSL + explicit SSH key.
$wsl = "C:\Windows\System32\wsl.exe"
$git = "/usr/bin/git"
$ssh = "/usr/bin/ssh"
$repo = "/mnt/d/projects/statistics_harness/statistics_harness"
$key = "/home/ninjra/.ssh/id_ed25519"

# Ensure SSH remote.
& $wsl $git -C $repo remote set-url origin git@github.com:ninjra/statistics_harness.git

# Push with explicit key.
& $wsl env GIT_SSH_COMMAND="$ssh -i $key -o IdentitiesOnly=yes" $git -C $repo push origin HEAD
