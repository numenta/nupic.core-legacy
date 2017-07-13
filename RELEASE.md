# Release Process

1. Select a green build in Bamboo that has the changes that you want to release
2. Ensure that this build has the correct version in the VERSION file
3. Create a Bamboo release with version matching the VERSION file in the repo
4. Deploy the release in Bamboo. This will:
    - Validate that the Bamboo release number matches the wheel versions.
    - Push the wheels to PyPI
    - Push the source archive to S3
    - Create a Github release
    - Tag the repo commit with the version
    - Bump the VERSION file to the next bugfix release number
