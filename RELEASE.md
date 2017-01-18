# Release Process

1. Send an announcement that a release is underway to the committer's lounge on
discourse.numenta.org and ask reviewers not to merge PRs in nupic.core until
you're done with the release.
2. Create a PR that includes:
    - Release notes added to CHANGELOG.md
    - Change to the VERSION file so it matches the intended release version
3. Wait for the PR to be approved and merged, and the Bamboo build to complete
successfully, along with the corresponding AppVeyor and Travis builds.
4. Fetch the additional nupic.core artifacts (from AppVeyor and Travis) in Bamboo via the
run button at top, selecting "Run stage 'Stage Release'".
5. Create a "release" in Bamboo with a version matching the intended release
version
6. Deploy the release in Bamboo. This will:
    - Validate that the Bamboo release number matches the wheel versions.
    - Push the wheels to PyPI
7. Create a new Github "Release" at https://github.com/numenta/nupic.core/releases/new
    - Along with the creation of the release, there is an option to create a git tag with the release. Name it "X.Y.Z" and point it to the commit SHA for the merged PR described in #2 above.
    - Release title should be "X.Y.Z"
    - Release description should be the latest changelog
8. Create a PR that includes the following, and wait for this PR to be approved and merged:
    - Change to the VERSION file with the next expected release and ".dev0" as suffix. For example, if the release was 0.5.0, then change it to 0.5.1.dev0
9. Send an announcement to the committer's lounge on discourse.numenta.org that the release is complete.