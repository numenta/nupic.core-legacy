<?xml version="1.0" encoding="UTF-8" ?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html"/>
    <xsl:template match="/">
        <xsl:text disable-output-escaping="yes"><![CDATA[<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">]]></xsl:text>
        <html>
            <head>
                <title>
                    Xunit Test Results - For: <xsl:value-of select="@name"/>
                </title>
                <style type="text/css">

                    .testcases {
                       border: 1px solid black;
                       padding: 10px;
                     }

                    .filters {
                        margin: 10px 0 0 0;
                     }

                    .failed {
                        display:block;
                    }
            
                   .failed a {
                        color: red;
                        font-weight: bold;
                    }

                    .passed {
                        display: block;
                    }

                   .passed a {
                        color: green;
                        font-weight: bold;
                    }
                
                    .error-message {
                        border: 1px solid red;
                    }        
    
                
                </style>
                <script src="http://code.jquery.com/jquery-1.11.1.min.js" type="text/javascript"></script>
                <script type="text/javascript">
                    $(document).ready(function() {
                        $(".filter").change(function(event) {
                            var self = $(this);
                            $('.filter option:selected').each(function() {
                                $('.testcases div').show();
                                if ( self.val() != 'all' ) {
                                    $('.testcases > div[class != "' + $(this).val() + '"][class!="error-message"]').each(function() {
                                        $(this).hide();
                                    });
                                }
                            });
                        }); 
                    });
                </script>
            </head>
            <body>
                <xsl:apply-templates/>
            </body>
        </html>
    </xsl:template>

    <xsl:template match="assemblies">
        <xsl:apply-templates/>
    </xsl:template>

    <xsl:template match="testsuite">

        <h3 class="divided">
            <b>Results for <xsl:value-of select="@name"/></b>
        </h3>

        <div class="container">
            Tests run: <a href="#all"><b><xsl:value-of select="@tests"/></b></a>
            Failures: <a href="#failures"><b><xsl:value-of select="@failures"/></b></a>,
            Skipped: <a href="#skipped"><b><xsl:value-of select="@skip"/></b></a>,
            Errors: <a href="#errors"><b><xsl:value-of select="@errors"/></b></a>

            <div class="filters">
                <label for="filter">Filter Test cases:</label>
                <select id="filter" class="filter">
                    <option name="all" value="all" selected="true">All</option>
                    <option name="passed" value="passed">Passed</option>
                    <option name="failed" value="failed">Failed</option>
                </select>
            </div>
            <hr></hr>
        </div>

        <div class="testcases">
        <xsl:for-each select="testcase">
            <xsl:choose>
                <xsl:when test="not(error) and not(failure)">
                    <div class="passed">    
                        <pre><b>Class:</b> <xsl:value-of select="@classname"/></pre>
                        <pre><b>Test Name:</b> <xsl:value-of select="@name"/></pre>
                        <pre><b>Running Time:</b> <xsl:value-of select="@time"/></pre>
                        <pre><b>State:</b><a>Passed</a></pre>
                        <xsl:for-each select="*">
                            <xsl:if test="@message">
                                <div class="error-message">
                                    <pre><a>>>>Error Message</a></pre>
                                    <xsl:value-of select="@message"/>
                                </div>
                            </xsl:if>
                        </xsl:for-each>
                    <hr></hr>                 
                   </div>
                </xsl:when>
                <xsl:otherwise>
                    <div class="failed">    
                        <pre><b>Class:</b> <xsl:value-of select="@classname"/></pre>
                        <pre><b>Test Name:</b> <xsl:value-of select="@name"/></pre>
                        <pre><b>Running Time:</b> <xsl:value-of select="@time"/></pre>
                        <pre><b>State:</b><a>Failed</a></pre>
                        <xsl:for-each select="error">
                            <div class="error-message">
                                <pre><a>>>>Error Message</a></pre>
                            <xsl:value-of select="@message"/>
                            </div>
                        </xsl:for-each>
                        <xsl:for-each select="failure">
                            <div class="failure-message">
                                <pre><a>>>>Failure Message</a></pre>
                            <xsl:value-of select="@message"/>
                            </div>
                        </xsl:for-each>
                    <hr></hr>                    
                    </div>
                </xsl:otherwise>
            </xsl:choose>
        </xsl:for-each>
        </div>

    </xsl:template>
</xsl:stylesheet>
