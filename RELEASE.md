# Release Process

The htm.core repository uses GitHub flow 
https://guides.github.com/introduction/flow/ style development.
Contributers to htm.core should be familiar with this Git strategy.

We implement Semantic Versioning (i.e. MAJOR.MINOR.PATCH )
See https://semver.org for a detailed definition but in gerneral
it means Bug fixes not affecting the API increment the PATCH version, 
backwards compatible API additions/changes (new features) increment the 
MINOR version, and backwards incompatible API changes (breaking changes) 
increment the MAJOR version. 

For this repository, the master is also the development branch so start your PR (push request) project by
creating a new branch off of the master.  When it works locally, push your PR
to back to GitHub where it is processed by three CIs.  When all three CIs pass all 
requirements (build works, unit tests, and approving reviews)
it can be merged into the master. 

Every merge to into the master is a release.  There are no pre-releases or beta releases.
The main rule of mainline development is that the master 
is always in a state that it could be deployed to production. This means that PRs 
should not be merged until they are ready to go out.



So increment the:

1. MAJOR version when you make API breaking changes in a PR,
2. MINOR version when you add major new features in a PR,
Otherwise the PATCH version will be automatically incremented during the 
merge of your PR into the master.

##  The steps

 * MAJOR release: While working on a branch that will be an API breaking change, use `+semver: major` as the first 
part of the commit message.  If there are multiple commits for a branch, only one needs this tag.
This will increment the MAJOR version and set MINOR and PATCH to 0 when this branch is merged into the master.
The next version can also be set manually by pushing a tag to the master.  This overrides all computed version numbers.
Over time the version calculation will slow down because it must search the git logs for commits, 
so it is recommended that a major release be declared using a tag rather than using `+semver: major` in the commit message.


 *  MINOR release: While working on a branch that will be a major new feature, use `semver: minor` as the first part of the
commit message.  This will increment the MINOR version and set PATCH to 0 when merged into the master.

 * PATCH release: Every merge to master is a release so if no semver flag is found in a commit on the branch being merged
the PATCH number is incremented automatically.


If you do not want GitVersion to treat a commit or a pull request as a release and 
increment the version you can use +semver: none or +semver: skip in a commit message to skip incrementing for that commit.
Normally we will not do this.

:-) 
