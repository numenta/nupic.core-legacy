

# setup the distr directory
set(distr ${CMAKE_INSTALL_PREFIX}/distr/src)
file(MAKE_DIRECTORY ${distr})
file(COPY ./src/. DESTINATION ${distr} PATTERN *)
message(STATUS "COPY py/src TO ${distr}")

