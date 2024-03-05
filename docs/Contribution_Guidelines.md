# Contributing to the Open-Sora Plan Community

The Open-Sora Plan open-source community is a collaborative initiative driven by the community, emphasizing a commitment to being free and void of exploitation. Organized spontaneously by community members, we invite you to contribute to the Open-Sora Plan open-source community and help elevate it to new heights!

## Submitting a Pull Request (PR)

As a contributor, before submitting your request, kindly follow these guidelines:

1. Start by checking the [Open-Sora Plan GitHub](https://github.com/PKU-YuanGroup/Open-Sora-Plan/pulls) to see if there are any open or closed pull requests related to your intended submission. Avoid duplicating existing work.

2. [Fork](https://github.com/PKU-YuanGroup/Open-Sora-Plan/fork) the [open-sora plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) repository and download your forked repository to your local machine.

   ```bash
   git clone [your-forked-repository-url]
   ```

3. Add the original Open-Sora Plan repository as a remote to sync with the latest updates:

   ```bash
   git remote add upstream https://github.com/PKU-YuanGroup/Open-Sora-Plan
   ```

4. Sync the code from the main repository to your local machine, and then push it back to your forked remote repository.

   ```
   # Pull the latest code from the upstream branch
   git fetch upstream
   
   # Switch to the main branch
   git checkout main
   
   # Merge the updates from the upstream branch into main, synchronizing the local main branch with the upstream
   git merge upstream/main
   
   # Additionally, sync the local main branch to the remote branch of your forked repository
   git push origin main
   ```


   > Note: Sync the code from the main repository before each submission.

5. Create a branch in your forked repository for your changes, ensuring the branch name is meaningful.

   ```bash
   git checkout -b my-docs-branch main
   ```

6. While making modifications and committing changes, adhere to our [Commit Message Format](#Commit-Message-Format).

   ```bash
   git commit -m "[docs]: xxxx"
   ```

7. Push your changes to your GitHub repository.

   ```bash
   git push origin my-docs-branch
   ```

8. Submit a pull request to `Open-Sora-Plan:main` on the GitHub repository page.

## Commit Message Format

Commit messages must include both `<type>` and `<summary>` sections.

```bash
[<type>]: <summary>
  │        │
  │        └─⫸ Briefly describe your changes, without ending with a period.
  │
  └─⫸ Commit Type: |docs|feat|fix|refactor|
```

### Type 

* **docs**: Modify or add documents.
* **feat**: Introduce a new feature.
* **fix**: Fix a bug.
* **refactor**: Restructure code, excluding new features or bug fixes.

### Summary

Describe modifications in English, without ending with a period.

> e.g., git commit -m "[docs]: add a contributing.md file"

This guideline is borrowed by [minisora](https://github.com/mini-sora/minisora). We sincerely appreciate MiniSora authors for their awesome templates. 
